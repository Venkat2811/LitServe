# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import contextlib
import copy
import inspect
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
import uuid
import warnings
from collections import deque, defaultdict
from contextlib import asynccontextmanager
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from starlette.formparsers import MultiPartParser
from starlette.middleware.gzip import GZipMiddleware

from litserve import LitAPI
from litserve.callbacks.base import Callback, CallbackRunner, EventTypes
from litserve.connector import _Connector
from litserve.loggers import Logger, _LoggerConnector
from litserve.loops import LitLoop, get_default_loop, inference_worker, simple_unified_loop
from litserve.loops.parallel_loops import decode_loop, preprocess_loop, predict_loop, encode_loop
from litserve.middlewares import MaxSizeMiddleware, RequestCountMiddleware
from litserve.python_client import client_template
from litserve.specs.base import LitSpec
from litserve.transport.base import MessageTransport
from litserve.transport.factory import TransportConfig, create_transport_from_config
from litserve.utils import LitAPIStatus, WorkerSetupStatus, call_after_stream

# Set multiprocessing start method to 'spawn' for better cross-platform compatibility
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')
mp.allow_connection_pickling()

logger = logging.getLogger(__name__)

# if defined, it will require clients to auth with X-API-Key in the header
LIT_SERVER_API_KEY = os.environ.get("LIT_SERVER_API_KEY")

# FastAPI writes form files to disk over 1MB by default, which prevents serialization by multiprocessing
MultiPartParser.max_file_size = sys.maxsize
# renamed in PR: https://github.com/encode/starlette/pull/2780
MultiPartParser.spool_max_size = sys.maxsize


def no_auth():
    pass


def api_key_auth(x_api_key: str = Depends(APIKeyHeader(name="X-API-Key"))):
    if x_api_key != LIT_SERVER_API_KEY:
        raise HTTPException(
            status_code=401, detail="Invalid API Key. Check that you are passing a correct 'X-API-Key' in your header."
        )


async def response_queue_to_buffer(
    transport: MessageTransport,
    response_buffer: Dict[str, Union[Tuple[deque, asyncio.Event], asyncio.Event]],
    stream: bool,
    consumer_id: int = 0,
):
    if stream:
        while True:
            try:
                result = await transport.areceive(consumer_id)
                if result is None:
                    continue

                uid, response = result
                stream_response_buffer, event = response_buffer[uid]
                stream_response_buffer.append(response)
                event.set()
            except asyncio.CancelledError:
                logger.debug("Response queue to buffer task was cancelled")
                break
            except Exception as e:
                logger.error(f"Error in response_queue_to_buffer: {e}")
                break

    else:
        while True:
            try:
                result = await transport.areceive(consumer_id)
                if result is None:
                    continue

                uid, response = result
                event = response_buffer.pop(uid)
                response_buffer[uid] = response
                event.set()
            except asyncio.CancelledError:
                logger.debug("Response queue to buffer task was cancelled")
                break
            except Exception as e:
                logger.error(f"Error in response_queue_to_buffer: {e}")
                break


def is_method_overridden(instance, cls, method_name):
    instance_method = getattr(instance, method_name, None)
    base_method = getattr(cls, method_name, None)
    return callable(instance_method) and instance_method.__code__ is not base_method.__code__


class LitServer:
    def __init__(
        self,
        lit_api: LitAPI,
        accelerator: str = "auto",
        devices: Union[str, int] = "auto",
        workers_per_device: int = 1,
        preprocess_workers_per_device: int = 1,
        execution_mode: str = "default",
        timeout: Union[float, bool] = 30,
        max_batch_size: int = 1,
        batch_timeout: float = 0.0,
        api_path: str = "/predict",
        healthcheck_path: str = "/health",
        info_path: str = "/info",
        model_metadata: Optional[dict] = None,
        stream: bool = False,
        spec: Optional[LitSpec] = None,
        max_payload_size=None,
        track_requests: bool = False,
        loop: Optional[Union[str, LitLoop]] = "auto",
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        middlewares: Optional[list[Union[Callable, tuple[Callable, dict]]]] = None,
        loggers: Optional[Union[Logger, List[Logger]]] = None,
        fast_queue: bool = False,
    ):
        # Create manager with explicit context for better cross-platform compatibility
        ctx = mp.get_context('spawn')
        self._manager = ctx.Manager()
        # Give the manager time to initialize fully
        time.sleep(0.5)

        """Initialize a LitServer instance.

        Args:
            lit_api: The API instance that handles requests and responses.
            accelerator: Type of hardware to use, like 'cpu', 'cuda', or 'mps'. 'auto' selects the best available.
            devices: Number of devices to use, or 'auto' to select automatically.
            workers_per_device: Number of inference worker processes per device.
            preprocess_workers_per_device: Number of preprocessing worker processes per device (used in 'full_parallel' mode).
            execution_mode: Execution strategy ('default' or 'full_parallel').
            timeout: Maximum time to wait for a request to complete. Set to False for no timeout.
            max_batch_size: Maximum number of requests to process in a batch.
            batch_timeout: Maximum time to wait for a batch to fill before processing.
            api_path: URL path for the prediction endpoint.
            healthcheck_path: URL path for the health check endpoint.
            info_path: URL path for the server and model information endpoint.
            model_metadata: Metadata about the model, shown at the info endpoint.
            stream: Whether to enable streaming responses.
            spec: Specification for the API, such as OpenAISpec or custom specs.
            max_payload_size: Maximum size of request payloads.
            track_requests: Whether to track the number of active requests.
            loop: Inference loop to use, or 'auto' to select based on settings.
            callbacks: List of callback classes to execute at various stages.
            middlewares: List of middleware classes to apply to the server.
            loggers: List of loggers to use for recording server activity.
            fast_queue: Whether to use ZeroMQ for faster response handling.
        """
        if batch_timeout > timeout and timeout not in (False, -1):
            raise ValueError("batch_timeout must be less than timeout")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than 0")
        if isinstance(spec, LitSpec):
            stream = spec.stream

        if loop is None:
            loop = "auto"

        if isinstance(loop, str) and loop != "auto":
            raise ValueError("loop must be an instance of _BaseLoop or 'auto'")
        if loop == "auto":
            loop = get_default_loop(stream, max_batch_size)

        if middlewares is None:
            middlewares = []
        if not isinstance(middlewares, list):
            _msg = (
                "middlewares must be a list of tuples"
                " where each tuple contains a middleware and its arguments. For example:\n"
                "server = ls.LitServer(ls.test_examples.SimpleLitAPI(), "
                'middlewares=[(RequestIdMiddleware, {"length": 5})])'
            )
            raise ValueError(_msg)

        if not api_path.startswith("/"):
            raise ValueError(
                "api_path must start with '/'. "
                "Please provide a valid api path like '/predict', '/classify', or '/v1/predict'"
            )

        if not healthcheck_path.startswith("/"):
            raise ValueError(
                "healthcheck_path must start with '/'. "
                "Please provide a valid api path like '/health', '/healthcheck', or '/v1/health'"
            )

        if not info_path.startswith("/"):
            raise ValueError(
                "info_path must start with '/'. Please provide a valid api path like '/info', '/details', or '/v1/info'"
            )

        try:
            json.dumps(model_metadata)
        except (TypeError, ValueError):
            raise ValueError("model_metadata must be JSON serializable.")

        # Check if the batch and unbatch methods are overridden in the lit_api instance
        batch_overridden = lit_api.batch.__code__ is not LitAPI.batch.__code__
        unbatch_overridden = lit_api.unbatch.__code__ is not LitAPI.unbatch.__code__

        if batch_overridden and unbatch_overridden and max_batch_size == 1:
            warnings.warn(
                "The LitServer has both batch and unbatch methods implemented, "
                "but the max_batch_size parameter was not set."
            )

        if sys.platform == "win32" and fast_queue:
            warnings.warn("ZMQ is not supported on Windows with LitServe. Disabling ZMQ.")
            fast_queue = False

        if execution_mode not in ["default", "full_parallel"]:
            raise ValueError("execution_mode must be either 'default' or 'full_parallel'")
        self.execution_mode = execution_mode
        self.preprocess_workers_per_device = preprocess_workers_per_device

        # Check if preprocess method is overridden
        self.preprocess_overridden = lit_api.preprocess.__code__ is not LitAPI.preprocess.__code__

        # Determine if separate preprocessing workers are needed
        self.use_preprocess_workers = (
            self.execution_mode == "full_parallel" and self.preprocess_overridden
        )

        self._loop: LitLoop = loop
        self.api_path = api_path
        self.healthcheck_path = healthcheck_path
        self.info_path = info_path
        self.track_requests = track_requests
        self.timeout = timeout
        lit_api.stream = stream
        lit_api.request_timeout = self.timeout
        lit_api.pre_setup(max_batch_size, spec=spec)
        self._loop.pre_setup(lit_api, spec=spec)
        self.app = FastAPI(lifespan=self.lifespan)
        self.app.response_queue_id = None
        self.response_queue_id = None
        self.response_buffer = {}
        # gzip does not play nicely with streaming, see https://github.com/tiangolo/fastapi/discussions/8448
        if not stream:
            middlewares.append((GZipMiddleware, {"minimum_size": 1000}))
        if max_payload_size is not None:
            middlewares.append((MaxSizeMiddleware, {"max_size": max_payload_size}))
        self.active_counters: List[mp.Value] = []
        self.middlewares = middlewares
        self._logger_connector = _LoggerConnector(self, loggers)
        self.logger_queue = None
        self.lit_api = lit_api
        self.lit_spec = spec
        self.workers_per_device = workers_per_device
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.stream = stream
        self.max_payload_size = max_payload_size
        self.model_metadata = model_metadata
        self._connector = _Connector(accelerator=accelerator, devices=devices)
        self._callback_runner = CallbackRunner(callbacks)
        self.use_zmq = fast_queue
        self.transport_config = None

        specs = spec if spec is not None else []
        self._specs = specs if isinstance(specs, Sequence) else [specs]

        decode_request_signature = inspect.signature(lit_api.decode_request)
        encode_response_signature = inspect.signature(lit_api.encode_response)

        self.request_type = decode_request_signature.parameters["request"].annotation
        if self.request_type == decode_request_signature.empty:
            self.request_type = Request

        self.response_type = encode_response_signature.return_annotation
        if self.response_type == encode_response_signature.empty:
            self.response_type = Response

        accelerator = self._connector.accelerator
        devices = self._connector.devices
        if accelerator == "cpu":
            self.devices = [accelerator]
        elif accelerator in ["cuda", "mps"]:
            device_list = devices
            if isinstance(devices, int):
                device_list = range(devices)
            self.devices = [self.device_identifiers(accelerator, device) for device in device_list]

        self.inference_workers = self.devices * self.workers_per_device
        self.transport_config = TransportConfig(transport_config="zmq" if self.use_zmq else "mp")
        self.num_decode_workers = 1
        self.num_preprocess_workers = preprocess_workers_per_device

        self.has_preprocess = is_method_overridden(self.lit_api, LitAPI, 'preprocess')

        # Validate mode vs preprocess implementation
        # if self.execution_mode == "full_parallel" and not self.has_preprocess:
        #     warnings.warn(
        #         "execution_mode='full_parallel' selected, but 'preprocess' method is not overridden in LitAPI. "
        #         "The server will run in a Decode -> Predict -> Encode parallel pipeline."
        #     )

        logger.info(f"Launched a total of {len(self.inference_workers)} workers.")
        logger.debug(f"[INIT] After configure_workers: self.num_encode_workers = {getattr(self, 'num_encode_workers', 'Not Set')}")
        # return manager, litserve_workers, worker_ready_queue # REMOVE THIS LINE

    def launch_inference_worker(self, num_api_servers: int, worker_ready_queue=None):
        """Launches worker processes based on the configured execution mode."""
        logger.info("Inside launch_inference_worker")
        ctx = mp.get_context("spawn") # Use spawn context

        # Create manager using the spawn context
        try:
            if self._manager is None:
                 self._manager = ctx.Manager()
        except Exception as e:
            logger.error(f"Failed to create Manager: {e}", exc_info=True)
            raise
        manager = self._manager

        self.transport_config.num_consumers = num_api_servers
        self.transport_config.manager = manager # Assign the correct manager
        logger.debug(f"[LAUNCH] Assigned manager {id(manager)} to transport_config.")

        try:
            self._transport = create_transport_from_config(self.transport_config) # Main response transport
        except Exception as e:
            logger.error(f"Failed to create transport: {e}", exc_info=True)
            raise

        self.workers_setup_status = manager.dict()
        self.request_queue = manager.Queue() # Main input queue from API servers

        if self._logger_connector._loggers:
            self.logger_queue = manager.Queue()

        self._logger_connector.run(self)

        for spec in self._specs:
            logging.debug(f"shallow copy for Server is created for spec {spec}")
            server_copy = copy.copy(self)
            del server_copy.app, server_copy.transport_config
            spec.setup(server_copy)

        process_list = []
        worker_id_counter = 0
        # --- Readiness Queue ---
        if worker_ready_queue is None:
            worker_ready_queue = manager.Queue()
        self.worker_ready_queue = worker_ready_queue # Store for run() method
        logger.debug(f"[LAUNCH] Using worker_ready_queue (proxy: {worker_ready_queue})")

        if self.execution_mode == "full_parallel":
            logger.info("Launching workers in 'full_parallel' mode.")
            # Create intermediate queues using the manager
            decoded_queue = manager.Queue()
            preprocess_queue = manager.Queue() if hasattr(self.lit_api, "preprocess") else None
            predict_output_queue = manager.Queue()

            # --- 1. Launch Decode Workers --- (Run on CPU)
            num_decode_workers = self.num_decode_workers
            logger.info(f"Launching {num_decode_workers} decode workers.")
            for i in range(num_decode_workers):
                worker_id = worker_id_counter
                process = ctx.Process( # Use ctx.Process
                    target=decode_loop,
                    args=(
                        type(self.lit_api),          # Pass API type
                        self.lit_spec,
                        self.request_queue, # Input: Main request queue
                        decoded_queue,      # Output: Decoded data queue
                        worker_id,
                        worker_ready_queue, # Pass only the queue proxy
                    ),
                    name=f"DecodeWorker-{worker_id}"
                )
                process.start()
                process_list.append(process)
                worker_id_counter += 1

            # --- 2. Launch Preprocess Workers (Optional) --- (Run on CPU)
            if hasattr(self.lit_api, "preprocess"):
                num_preprocess_workers = self.num_preprocess_workers
                logger.info(f"Launching {num_preprocess_workers} preprocess workers.")
                for i in range(num_preprocess_workers):
                    worker_id = worker_id_counter
                    process = ctx.Process( # Use ctx.Process
                        target=preprocess_loop,
                        args=(
                            type(self.lit_api),       # Pass API type
                            self.lit_spec,
                            decoded_queue,       # Input: Decoded data queue
                            preprocess_queue,    # Output: Preprocessed data queue
                            worker_id,
                            worker_ready_queue,  # Pass only the queue proxy
                        ),
                        name=f"PreprocessWorker-{worker_id}"
                    )
                    process.start()
                    process_list.append(process)
                    worker_id_counter += 1

            # --- 3. Launch Predict Workers --- (Run on specified devices)
            logger.info(f"Launching {len(self.inference_workers)} predict workers.")
            predict_input_queue = preprocess_queue if preprocess_queue else decoded_queue
            for i, device_tuple in enumerate(self.inference_workers):
                device = device_tuple[0] # Get the actual device string
                worker_id = worker_id_counter
                process = ctx.Process( # Use ctx.Process
                    target=predict_loop,
                    args=(
                        type(self.lit_api),          # Pass API type
                        self.lit_spec,
                        device,
                        predict_input_queue,    # Input: Preprocessed or Decoded queue
                        predict_output_queue,   # Output: Prediction results queue
                        worker_id,
                        self.max_batch_size,
                        self.batch_timeout,
                        worker_ready_queue,     # Pass only the queue proxy
                    ),
                    name=f"PredictWorker-{worker_id}-{device}"
                )
                process.start()
                process_list.append(process)
                worker_id_counter += 1

            # --- 4. Launch Encode Workers --- (Run on CPU)
            # Reuse num_decode_workers for encode workers for simplicity
            num_encode_workers = self.num_decode_workers
            logger.info(f"Launching {num_encode_workers} encode workers.")
            for i in range(num_encode_workers):
                worker_id = worker_id_counter
                response_input_queue = self._transport._queues[i] # Match encode worker index to transport queue index
                process = ctx.Process( # Use ctx.Process
                    target=encode_loop,
                    args=(
                        type(self.lit_api),        # Pass API type
                        self.lit_spec,
                        predict_output_queue, # Input: Prediction results queue
                        response_input_queue,      # Output: Specific output queue
                        worker_id,
                        worker_ready_queue,        # Pass only the queue proxy
                    ),
                    name=f"EncodeWorker-{worker_id}"
                )
                process.start()
                process_list.append(process)
                worker_id_counter += 1
        else:
            # Default mode worker launch
            logger.info(f"Launching {self.inference_workers} workers in 'default' mode.")
            num_default_workers = 0
            for device_tuple in self.inference_workers:
                device = device_tuple[0]
                worker_id = num_default_workers # Simple counter for default
                process = ctx.Process( # Use ctx.Process
                    target=simple_unified_loop, # Assuming this exists and is updated
                    args=(
                        self.lit_api, # Pass API instance
                        self.lit_spec,
                        device,
                        worker_id,
                        self.request_queue,      # Input queue
                        self._transport,         # Output transport
                        self.max_batch_size,
                        self.batch_timeout,
                        self.stream,
                        self.workers_setup_status, # Keep status dict for default
                    ),
                    name=f"DefaultWorker-{worker_id}-{device}"
                )
                process.start()
                process_list.append(process)
                num_default_workers += 1
            logger.info(f"Launched {num_default_workers} default workers.")
        return manager, process_list # Don't return queue anymore

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        loop = asyncio.get_running_loop()

        if not hasattr(self, "_transport") or not self._transport:
            raise RuntimeError(
                "Response queues have not been initialized. "
                "Please make sure to call the 'launch_inference_worker' method of "
                "the LitServer class to initialize the response queues."
            )

        transport = self._transport
        future = response_queue_to_buffer(
            transport,
            self.response_buffer,
            self.stream,
            app.response_queue_id,
        )
        task = loop.create_task(future, name=f"response_queue_to_buffer-{app.response_queue_id}")

        try:
            yield
        finally:
            self._callback_runner.trigger_event(EventTypes.ON_SERVER_END, litserver=self)

            # Cancel the task
            task.cancel()

            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError, Exception):
                await asyncio.wait_for(task, timeout=1.0)

    def device_identifiers(self, accelerator, device):
        if isinstance(device, Sequence):
            return [f"{accelerator}:{el}" for el in device]
        return [f"{accelerator}:{device}"]

    async def data_streamer(self, q: deque, data_available: asyncio.Event, send_status: bool = False):
        while True:
            await data_available.wait()
            while len(q) > 0:
                data, status = q.popleft()
                if status == LitAPIStatus.FINISH_STREAMING:
                    return

                if status == LitAPIStatus.ERROR:
                    logger.error(
                        "Error occurred while streaming outputs from the inference worker. "
                        "Please check the above traceback."
                    )
                    if send_status:
                        yield data, status
                    return
                if send_status:
                    yield data, status
                else:
                    yield data
            data_available.clear()

    @property
    def active_requests(self):
        if self.track_requests and self.active_counters:
            return sum(counter.value for counter in self.active_counters)
        return None

    def register_endpoints(self):
        """Register endpoint routes for the FastAPI app and setup middlewares."""
        self._callback_runner.trigger_event(EventTypes.ON_SERVER_START, litserver=self)
        workers_ready = False

        @self.app.get("/", dependencies=[Depends(self.setup_auth())])
        async def index(request: Request) -> Response:
            return Response(content="litserve running")

        @self.app.get(self.healthcheck_path, dependencies=[Depends(self.setup_auth())])
        async def health(request: Request) -> Response:
            nonlocal workers_ready
            if not workers_ready:
                # For full_parallel mode, be more lenient with worker readiness
                if self.execution_mode == "full_parallel":
                    # In full_parallel mode, consider healthy if any workers are ready
                    workers_ready = any(v == WorkerSetupStatus.READY for v in self.workers_setup_status.values())
                    if workers_ready:
                        logger.info(f"Health check: At least one worker is ready in full_parallel mode.")
                else:
                    # For default mode, all workers must be ready
                    workers_ready = all(v == WorkerSetupStatus.READY for v in self.workers_setup_status.values())

            lit_api_health_status = self.lit_api.health()
            if workers_ready and lit_api_health_status:
                return Response(content="ok", status_code=200)

            return Response(content="not ready", status_code=503)

        @self.app.get(self.info_path, dependencies=[Depends(self.setup_auth())])
        async def info(request: Request) -> Response:
            return JSONResponse(
                content={
                    "model": self.model_metadata,
                    "server": {
                        "devices": self.devices,
                        "workers_per_device": self.workers_per_device,
                        "timeout": self.timeout,
                        "max_batch_size": self.max_batch_size,
                        "batch_timeout": self.batch_timeout,
                        "stream": self.stream,
                        "max_payload_size": self.max_payload_size,
                        "track_requests": self.track_requests,
                    },
                }
            )

        async def predict(request: self.request_type) -> self.response_type:
            self._callback_runner.trigger_event(
                EventTypes.ON_REQUEST,
                active_requests=self.active_requests,
                litserver=self,
            )
            response_queue_id = self.app.response_queue_id
            uid = uuid.uuid4()
            event = asyncio.Event()
            self.response_buffer[uid] = event
            logger.debug(f"Received request uid={uid}")

            payload = request
            if self.request_type == Request:
                if request.headers["Content-Type"] == "application/x-www-form-urlencoded" or request.headers[
                    "Content-Type"
                ].startswith("multipart/form-data"):
                    payload = await request.form()
                else:
                    payload = await request.json()

            self.request_queue.put((response_queue_id, uid, time.monotonic(), payload))

            await event.wait()
            response, status = self.response_buffer.pop(uid)
            if status == LitAPIStatus.ERROR and isinstance(response, HTTPException):
                logger.error("Error in request: %s", response)
                raise response
            if status == LitAPIStatus.ERROR:
                logger.error("Error in request: %s", response)
                raise HTTPException(status_code=500)
            self._callback_runner.trigger_event(EventTypes.ON_RESPONSE, litserver=self)
            return response

        async def stream_predict(request: self.request_type) -> self.response_type:
            self._callback_runner.trigger_event(
                EventTypes.ON_REQUEST,
                active_requests=self.active_requests,
                litserver=self,
            )
            response_queue_id = self.app.response_queue_id
            uid = uuid.uuid4()
            event = asyncio.Event()
            q = deque()
            self.response_buffer[uid] = (q, event)
            logger.debug(f"Received request uid={uid}")

            payload = request
            if self.request_type == Request:
                payload = await request.json()
            self.request_queue.put((response_queue_id, uid, time.monotonic(), payload))

            response = call_after_stream(
                self.data_streamer(q, data_available=event),
                self._callback_runner.trigger_event,
                EventTypes.ON_RESPONSE,
                litserver=self,
            )
            return StreamingResponse(response)

        if not self._specs:
            stream = self.lit_api.stream
            # In the future we might want to differentiate endpoints for streaming vs non-streaming
            # For now we allow either one or the other
            endpoint = self.api_path
            methods = ["POST"]
            self.app.add_api_route(
                endpoint,
                stream_predict if stream else predict,
                methods=methods,
                dependencies=[Depends(self.setup_auth())],
            )

        for spec in self._specs:
            spec: LitSpec
            # TODO check that path is not clashing
            for path, endpoint, methods in spec.endpoints:
                self.app.add_api_route(
                    path, endpoint=endpoint, methods=methods, dependencies=[Depends(self.setup_auth())]
                )

        for middleware in self.middlewares:
            if isinstance(middleware, tuple):
                middleware, kwargs = middleware
                self.app.add_middleware(middleware, **kwargs)
            elif callable(middleware):
                self.app.add_middleware(middleware)

    def run(
        self,
        host: str = "0.0.0.0",
        port: Union[str, int] = 8000,
        num_api_servers: Optional[int] = None,
        log_level: str = "info",
        generate_client_file: bool = True,
        api_server_worker_type: Optional[str] = None,
        **kwargs,
    ):
        print(f"[{os.getpid()}] Entering LitServer.run for port {port}") # Diagnostic print

        logger.debug(f"[RUN] Start: self.num_encode_workers = {getattr(self, 'num_encode_workers', 'Not Set')}")

        if generate_client_file:
            LitServer.generate_client_file(port=port)

        port_msg = f"port must be a value from 1024 to 65535 but got {port}"
        try:
            port = int(port)
        except ValueError:
            raise ValueError(port_msg)

        if not (1024 <= port <= 65535):
            raise ValueError(port_msg)

        host_msg = f"host must be '0.0.0.0', '127.0.0.1', or '::' but got {host}"
        if host not in ["0.0.0.0", "127.0.0.1", "::"]:
            raise ValueError(host_msg)

        config = uvicorn.Config(app=self.app, host=host, port=port, log_level=log_level, **kwargs)
        sockets = [config.bind_socket()]

        if num_api_servers is None:
            num_api_servers = len(self.inference_workers)

        if num_api_servers < 1:
            raise ValueError("num_api_servers must be greater than 0")

        if sys.platform == "win32":
            warnings.warn(
                "Windows does not support forking. Using threads api_server_worker_type will be set to 'thread'"
            )
            api_server_worker_type = "thread"
        elif api_server_worker_type is None:
            api_server_worker_type = "process"

        servers = []  # Initialize at top for finally block
        manager = None
        try:
            print(f"[{os.getpid()}] Calling launch_inference_worker...") # Diagnostic print
            # launch_inference_worker should return the started manager and worker list
            manager = mp.Manager()
            self._manager = manager  # Keep the manager alive
            self.worker_ready_queue = manager.Queue()  # Initialize the worker ready queue
            litserve_workers = self.launch_inference_worker(num_api_servers, self.worker_ready_queue)

            # --- Calculate Expected Workers (AFTER launch_inference_worker) ---
            if self.execution_mode == "full_parallel":
                # Ensure worker count attributes are set before use
                self.num_encode_workers = getattr(self, 'num_encode_workers', self.num_decode_workers)
                self.num_predict_workers = getattr(self, 'num_predict_workers', len(self.inference_workers))
                # Calculate expected workers for parallel mode (attributes set by configure_workers)
                num_expected_workers = self.num_decode_workers + self.num_encode_workers
                if hasattr(self.lit_api, "preprocess"):
                    num_expected_workers += self.num_preprocess_workers
                num_expected_workers += self.num_predict_workers # predict workers always exist
            else:
                # Default mode: expected workers is the number of launched processes
                num_expected_workers = len(litserve_workers)

            logger.info(
                f"launch_inference_worker completed. Manager: {id(manager)}, Workers: {len(litserve_workers)}"
            )

            if self.execution_mode == "full_parallel":
                # Wait for all parallel workers to signal readiness via the queue (accessed via self)
                logger.info(f"Waiting for {num_expected_workers} workers to report ready via queue...")
                ready_signals_received = 0
                start_wait_time = time.monotonic()
                wait_timeout = 60.0 # Seconds to wait for workers

                if not hasattr(self, 'worker_ready_queue') or self.worker_ready_queue is None:
                     logger.error("Worker ready queue not initialized on server instance!")
                     raise RuntimeError("Internal server error: Readiness queue missing.")

                while ready_signals_received < num_expected_workers:
                    try:
                        # Wait for the next worker to signal readiness
                        ready_worker_id = self.worker_ready_queue.get(timeout=wait_timeout - (time.monotonic() - start_wait_time))
                        ready_signals_received += 1
                        logger.debug(f"Received ready signal from worker {ready_worker_id} ({ready_signals_received}/{num_expected_workers}).")
                    except queue.Empty:
                        logger.error(f"Timeout: Only received {ready_signals_received}/{num_expected_workers} ready signals after {wait_timeout} seconds.")
                        raise RuntimeError(f"Workers failed to start within the timeout period.")
                    except Exception as e:
                        logger.error(f"Error waiting for worker readiness queue: {e}", exc_info=True)
                        raise
                logger.info(f"All {num_expected_workers} workers reported ready.")
            else:
                # Optional: Add readiness check for default mode if needed
                # self.verify_worker_status() # Could call the old method here
                logger.info("Default mode: Skipping explicit worker readiness wait (handled internally or via status dict)." )


            # Start the Uvicorn server in the main process
            for response_queue_id in range(num_api_servers):
                self.app.response_queue_id = response_queue_id
                if self.lit_spec:
                    self.lit_spec.response_queue_id = response_queue_id
                app: FastAPI = copy.copy(self.app)

                self._prepare_app_run(app)

                config = uvicorn.Config(app=app, host="0.0.0.0", port=port, log_level=log_level, **kwargs)
                server = uvicorn.Server(config=config)
                if api_server_worker_type == "process":
                    ctx = mp.get_context("fork")
                    w = ctx.Process(target=server.run, args=(sockets,))
                elif api_server_worker_type == "thread":
                    w = threading.Thread(target=server.run, args=(sockets,))
                else:
                    raise ValueError("Invalid value for api_server_worker_type. Must be 'process' or 'thread'")
                w.start()
                servers.append(w)
            print(f"Swagger UI is available at http://0.0.0.0:{port}/docs")
            for s in servers:
                s.join()
        finally:
            print(f"[{os.getpid()}] Entering finally block for shutdown...")
            # Graceful shutdown sequence
            # Terminate any running Uvicorn servers
            if servers:
                for server in servers:
                    if server.is_alive():
                        print(f"[{os.getpid()}] Shutdown: Terminating Uvicorn server process/thread...")
                        # Use terminate for processes, interrupt might be needed for threads? Assuming terminate is sufficient.
                        if hasattr(server, 'terminate'):
                            server.terminate()
                        # Join the server process/thread after attempting termination
                        server.join(timeout=2) # Add a timeout
                        if server.is_alive():
                            print(f"!!! [{os.getpid()}] Warning: Uvicorn server did not terminate gracefully.")

            # Terminate LitServe inference workers BEFORE shutting down the manager
            if 'litserve_workers' in locals() and litserve_workers: # Check if list exists and is not empty
                print(f"[{os.getpid()}] Shutdown: Terminating {len(litserve_workers)} LitServe inference workers...")
                for worker in litserve_workers:
                    if worker.is_alive():
                        print(f"[{os.getpid()}] Terminating worker PID {worker.pid}...")
                        worker.terminate() # Send SIGTERM
                # Wait for workers to terminate
                for worker in litserve_workers:
                    print(f"[{os.getpid()}] Joining worker PID {worker.pid}...")
                    worker.join(timeout=2) # Wait for worker to exit, with timeout
                    if worker.is_alive():
                        print(f"!!! [{os.getpid()}] Warning: Worker PID {worker.pid} did not terminate gracefully.")

            # Now shut down the manager
            if manager is not None:
                print(f"[{os.getpid()}] Shutdown: Shutting down multiprocessing manager...")
                try:
                    manager.shutdown()
                except Exception as manager_shutdown_e:
                     print(f"!!! [{os.getpid()}] Error shutting down manager: {manager_shutdown_e}")

            print(f"[{os.getpid()}] Shutdown complete.")

    def _prepare_app_run(self, app: FastAPI):
        # Add middleware to count active requests
        active_counter = mp.Value("i", 0, lock=True)
        self.active_counters.append(active_counter)
        app.add_middleware(RequestCountMiddleware, active_counter=active_counter)

    def setup_auth(self):
        if hasattr(self.lit_api, "authorize") and callable(self.lit_api.authorize):
            return self.lit_api.authorize
        if LIT_SERVER_API_KEY:
            return api_key_auth
        return no_auth

    @staticmethod
    def generate_client_file(port: Union[str, int] = 8000):
        dest_path = os.path.join(os.getcwd(), "client.py")

        if os.path.exists(dest_path):
            logger.debug("client.py already exists in the current directory. Skipping generation.")
            return

        try:
            client_code = client_template.format(PORT=port)
            with open(dest_path, "w") as f:
                f.write(client_code)

        except Exception as e:
            logger.exception(f"Error copying file: {e}")

    def stop(self):
        if self._server:
            self._server.stop()
        # Add worker cleanup
        for process in self.worker_processes:
            process.terminate()
        self.worker_processes = []
