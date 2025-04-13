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
import logging
import time
from queue import Empty, Queue
from typing import Dict, Optional, Any

from fastapi import HTTPException

from litserve import LitAPI
from litserve.callbacks import CallbackRunner, EventTypes
from litserve.loops.base import DefaultLoop, _inject_context, collate_requests
from litserve.specs.base import LitSpec
from litserve.transport.base import MessageTransport
from litserve.utils import LitAPIStatus, PickleableHTTPException

logger = logging.getLogger(__name__)


class SingleLoop(DefaultLoop):
    def run_single_loop(
        self,
        lit_api: LitAPI,
        lit_spec: Optional[LitSpec],
        request_queue: Queue,
        transport: MessageTransport,
        callback_runner: CallbackRunner,
    ):
        while True:
            try:
                response_queue_id, uid, timestamp, x_enc = request_queue.get(timeout=1.0)
            except (Empty, ValueError):
                continue

            # Check for sentinel value used in tests to stop the loop
            if uid is None:
                break

            if (lit_api.request_timeout and lit_api.request_timeout != -1) and (
                time.monotonic() - timestamp > lit_api.request_timeout
            ):
                logger.error(
                    f"Request {uid} was waiting in the queue for too long ({lit_api.request_timeout} seconds) and "
                    "has been timed out. "
                    "You can adjust the timeout by providing the `timeout` argument to LitServe(..., timeout=30)."
                )
                self.put_response(
                    transport=transport,
                    response_queue_id=response_queue_id,
                    uid=uid,
                    response_data=(HTTPException(504, "Request timed out")),
                    status=LitAPIStatus.ERROR,
                )
                continue
            try:
                context = {}
                if hasattr(lit_spec, "populate_context"):
                    lit_spec.populate_context(context, x_enc)

                callback_runner.trigger_event(EventTypes.BEFORE_DECODE_REQUEST, lit_api=lit_api)
                x = _inject_context(
                    context,
                    lit_api.decode_request,
                    x_enc,
                )
                callback_runner.trigger_event(EventTypes.AFTER_DECODE_REQUEST, lit_api=lit_api)

                callback_runner.trigger_event(EventTypes.BEFORE_PREDICT, lit_api=lit_api)
                y = _inject_context(
                    context,
                    lit_api.predict,
                    x,
                )
                callback_runner.trigger_event(EventTypes.AFTER_PREDICT, lit_api=lit_api)

                callback_runner.trigger_event(EventTypes.BEFORE_ENCODE_RESPONSE, lit_api=lit_api)
                y_enc = _inject_context(
                    context,
                    lit_api.encode_response,
                    y,
                )
                callback_runner.trigger_event(EventTypes.AFTER_ENCODE_RESPONSE, lit_api=lit_api)
                self.put_response(
                    transport=transport,
                    response_queue_id=response_queue_id,
                    uid=uid,
                    response_data=y_enc,
                    status=LitAPIStatus.OK,
                )

            except HTTPException as e:
                self.put_response(
                    transport=transport,
                    response_queue_id=response_queue_id,
                    uid=uid,
                    response_data=PickleableHTTPException.from_exception(e),
                    status=LitAPIStatus.ERROR,
                )

            except Exception as e:
                logger.exception(
                    "LitAPI ran into an error while processing the request uid=%s.\n"
                    "Please check the error trace for more details.",
                    uid,
                )
                self.put_error_response(
                    transport=transport,
                    response_queue_id=response_queue_id,
                    uid=uid,
                    error=e,
                )

    def __call__(
        self,
        lit_api: LitAPI,
        lit_spec: Optional[LitSpec],
        device: str,
        worker_id: int,
        request_queue: Queue,
        transport: MessageTransport,
        max_batch_size: int,
        batch_timeout: float,
        stream: bool,
        workers_setup_status: Dict[int, str],
        callback_runner: CallbackRunner,
    ):
        self.run_single_loop(lit_api, lit_spec, request_queue, transport, callback_runner)


class BatchedLoop(DefaultLoop):
    def run_batched_loop(
        self,
        lit_api: LitAPI,
        lit_spec: LitSpec,
        request_queue: Queue,
        transport: MessageTransport,
        max_batch_size: int,
        batch_timeout: float,
        callback_runner: CallbackRunner,
    ):
        while True:
            batches, timed_out_uids, sentinel_found = collate_requests(
                lit_api,
                request_queue,
                max_batch_size,
                batch_timeout,
            )

            for response_queue_id, uid in timed_out_uids:
                logger.error(
                    f"Request {uid} was waiting in the queue for too long ({lit_api.request_timeout} seconds) and "
                    "has been timed out. "
                    "You can adjust the timeout by providing the `timeout` argument to LitServe(..., timeout=30)."
                )
                self.put_response(
                    transport, response_queue_id, uid, HTTPException(504, "Request timed out"), LitAPIStatus.ERROR
                )

            # Stop the loop if the sentinel was detected by collate_requests
            if sentinel_found:
                break

            if not batches:
                continue
            logger.debug(f"{len(batches)} batched requests received")
            response_queue_ids, uids, inputs = zip(*batches)
            num_inputs = len(inputs)
            try:
                contexts = [{} for _ in range(num_inputs)]
                if hasattr(lit_spec, "populate_context"):
                    for input, context in zip(inputs, contexts):
                        lit_spec.populate_context(context, input)

                callback_runner.trigger_event(EventTypes.BEFORE_DECODE_REQUEST, lit_api=lit_api)
                x = [
                    _inject_context(
                        context,
                        lit_api.decode_request,
                        input,
                    )
                    for input, context in zip(inputs, contexts)
                ]
                callback_runner.trigger_event(EventTypes.AFTER_DECODE_REQUEST, lit_api=lit_api)

                x = lit_api.batch(x)

                callback_runner.trigger_event(EventTypes.BEFORE_PREDICT, lit_api=lit_api)
                y = _inject_context(contexts, lit_api.predict, x)
                callback_runner.trigger_event(EventTypes.AFTER_PREDICT, lit_api=lit_api)

                outputs = lit_api.unbatch(y)

                if len(outputs) != num_inputs:
                    logger.error(
                        "LitAPI.predict/unbatch returned {len(outputs)} outputs, but expected {num_inputs}. "
                        "Please check the predict/unbatch method of the LitAPI implementation."
                    )
                    raise HTTPException(500, "Batch size mismatch")

                callback_runner.trigger_event(EventTypes.BEFORE_ENCODE_RESPONSE, lit_api=lit_api)
                y_enc_list = []
                for response_queue_id, y, uid, context in zip(response_queue_ids, outputs, uids, contexts):
                    y_enc = _inject_context(context, lit_api.encode_response, y)
                    y_enc_list.append((response_queue_id, uid, y_enc))
                callback_runner.trigger_event(EventTypes.AFTER_ENCODE_RESPONSE, lit_api=lit_api)

                for response_queue_id, uid, y_enc in y_enc_list:
                    self.put_response(transport, response_queue_id, uid, y_enc, LitAPIStatus.OK)

            except HTTPException as e:
                for response_queue_id, uid in zip(response_queue_ids, uids):
                    self.put_response(
                        transport,
                        response_queue_id,
                        uid,
                        PickleableHTTPException.from_exception(e),
                        LitAPIStatus.ERROR,
                    )

            except Exception as e:
                logger.exception(
                    "LitAPI ran into an error while processing the batched request.\n"
                    "Please check the error trace for more details."
                )
                for response_queue_id, uid in zip(response_queue_ids, uids):
                    self.put_error_response(transport, response_queue_id, uid, e)

    def __call__(
        self,
        lit_api: LitAPI,
        lit_spec: Optional[LitSpec],
        device: str,
        worker_id: int,
        request_queue: Queue,
        transport: MessageTransport,
        max_batch_size: int,
        batch_timeout: float,
        stream: bool,
        workers_setup_status: Dict[int, str],
        callback_runner: CallbackRunner,
    ):
        self.run_batched_loop(
            lit_api,
            lit_spec,
            request_queue,
            transport,
            max_batch_size,
            batch_timeout,
            callback_runner,
        )


def simple_unified_loop(
    lit_api: LitAPI,
    lit_spec: Optional[LitSpec], 
    device: str,
    worker_id: int,
    request_queue: Any,
    response_transport: Any,
    max_batch_size: int,
    batch_timeout: float,
    stream: bool, 
    setup_status: Dict,
    has_preprocess: bool,
):
    """Worker loop for 'simple' execution mode, running all steps sequentially with batching."""
    logger.info(f"Default worker {worker_id} starting on device {device}.")
    requires_batch = lit_api.batch.__code__ is not LitAPI.batch.__code__
    requires_unbatch = lit_api.unbatch.__code__ is not LitAPI.unbatch.__code__

    try:
        lit_api.setup(device)
        if _is_torch_available():
            torch.device(device)
            torch.cuda.empty_cache()
        setup_status[worker_id] = WorkerSetupStatus.READY
        logger.info(f"Default worker {worker_id} ready on device {device}.")
    except Exception as e:
        logger.error(f"Default worker {worker_id} setup failed: {e}", exc_info=True)
        setup_status[worker_id] = WorkerSetupStatus.FAILED
        return

    while True:
        try:
            # 1. Collate requests into batches
            batches, timed_out_uids = collate_requests(
                lit_api=lit_api,
                request_queue=request_queue,
                max_batch_size=max_batch_size,
                batch_timeout=batch_timeout,
            )

            # Handle timed-out requests
            for response_queue_id, uid, timestamp in timed_out_uids:
                logger.error(
                    f"Request {uid} timed out waiting in queue (timeout={lit_api.request_timeout}s). Worker {worker_id}."
                )
                try:
                    # Send timeout error back
                    error_payload = HTTPException(504, "Request timed out in queue")
                    response_transport.put((response_queue_id, uid, timestamp, error_payload))
                except Exception as send_e:
                    logger.error(f"Default worker {worker_id} failed to send timeout error for {uid}: {send_e}")

            if not batches:
                # No batches ready, continue loop (collate_requests handles waiting)
                continue

            logger.debug(f"Default worker {worker_id} processing batch of size {len(batches)}.")
            response_queue_ids, uids, timestamps, inputs_enc = zip(*batches)
            num_inputs = len(inputs_enc)
            outputs_enc = [None] * num_inputs # Placeholder for results or errors
            batch_processed_successfully = True

            try:
                # 2. Decode requests
                # TODO: Add context injection if LitSpec is passed and used
                decoded_inputs = [lit_api.decode_request(x_enc) for x_enc in inputs_enc]

                # 3. Preprocess (optional)
                if has_preprocess:
                    preprocessed_inputs = [lit_api.preprocess(x) for x in decoded_inputs]
                else:
                    preprocessed_inputs = decoded_inputs

                # 4. Batch for prediction
                if requires_batch:
                    batched_input_for_predict = lit_api.batch(preprocessed_inputs)
                else:
                    # If batch size is 1, or no batch method, pass single item or list
                    if max_batch_size == 1:
                        batched_input_for_predict = preprocessed_inputs[0]
                    else:
                        batched_input_for_predict = preprocessed_inputs

                # 5. Predict
                predict_start_time = time.monotonic()
                results = lit_api.predict(batched_input_for_predict)
                predict_duration = time.monotonic() - predict_start_time
                logger.debug(f"Default worker {worker_id} predict duration: {predict_duration:.4f}s for batch size {num_inputs}")

                # 6. Unbatch results
                if requires_unbatch:
                    unbatched_outputs = lit_api.unbatch(results)
                else:
                    if max_batch_size == 1:
                        unbatched_outputs = [results] # Ensure list
                    else:
                        if not isinstance(results, (list, tuple)) or len(results) != num_inputs:
                             logger.warning(
                                f"Default worker {worker_id}: Predict output count ({len(results) if isinstance(results, (list, tuple)) else 'N/A'}) does not match batch size ({num_inputs}). "
                                f"Ensure predict returns list/tuple or implement unbatch."
                            )
                             if isinstance(results, (list, tuple)):
                                 unbatched_outputs = results
                             else:
                                raise TypeError(f"Expected predict output to be list/tuple for batch size > 1, got {type(results)}")
                        else:
                            unbatched_outputs = results
            
                if len(unbatched_outputs) != num_inputs:
                     raise RuntimeError(f"Unbatched output count ({len(unbatched_outputs)}) does not match batch size ({num_inputs}).")

                # 7. Encode responses
                outputs_enc = [lit_api.encode_response(y) for y in unbatched_outputs]

            except Exception as e:
                batch_processed_successfully = False
                logger.error(f"Default worker {worker_id} failed processing batch: {e}", exc_info=True)
                # Fill outputs_enc with error payload for all items in the batch
                # TODO: Define a better error format
                error_payload = e if isinstance(e, HTTPException) else Exception(f"Processing failed in worker {worker_id}: {e}")
                outputs_enc = [error_payload] * num_inputs

            # 8. Send responses (or errors)
            for i in range(num_inputs):
                try:
                    response_payload = outputs_enc[i]
                    response_transport.put((response_queue_ids[i], uids[i], timestamps[i], response_payload))
                except Exception as send_e:
                     logger.error(f"Default worker {worker_id} failed to send response/error for {uids[i]}: {send_e}")
        except (KeyboardInterrupt, SystemExit):
            logger.info(f"Default worker {worker_id} received exit signal.")
            break
        except Exception as e:
             # Catch-all for unexpected errors outside batch processing (e.g., in collate_requests itself?) unlikely
             logger.error(f"Default worker {worker_id} encountered unexpected error in main loop: {e}", exc_info=True)
             time.sleep(1)

    logger.info(f"Default worker {worker_id} stopped.")
