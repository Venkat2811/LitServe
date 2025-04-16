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
import queue
import time
import multiprocessing as mp
import random
from typing import Any, Dict, Type, Optional
from multiprocessing.managers import RemoteError
from multiprocessing.synchronize import Event

from litserve.api import LitAPI
from litserve.specs.base import LitSpec
from litserve.utils import _is_torch_available, WorkerSetupStatus

if _is_torch_available():
    import torch

logger = logging.getLogger(__name__)


def decode_loop(
    api_cls: Type[LitAPI],
    lit_spec: Optional[LitSpec],
    input_queue: Any,  # Transport Queue (e.g., mp.Queue)
    output_queue: Any, # Transport Queue
    worker_id: int,
    ready_queue: Any,
    callback_runner=None
):
    """Worker loop dedicated to decoding requests."""
    # Instantiate the API object
    lit_api = api_cls() # Instantiate API
    logger.info(f"Decode worker {worker_id} starting.")
    # Initial setup and connection retry
    setup_attempts = 5
    setup_success = False
    for attempt in range(setup_attempts):
        try:
            # Perform minimal setup if needed, maybe device placement? For now, assume CPU
            # lit_api.setup("cpu") # Or a separate setup? Let's rely on predict_loop setup for now.
            logger.info(f"Decode worker {worker_id}: Initializing API...")
            lit_api.setup(device="cpu")  # Decode always runs on CPU
            logger.info(f"Decode worker {worker_id}: API setup complete.")
            ready_queue.put(worker_id) # Signal readiness via ready_queue
            setup_success = True
            break
        except (FileNotFoundError, AttributeError) as e: # Catch specific connection errors
            if "'ForkAwareLocal' object has no attribute 'connection'" not in str(e) and not isinstance(e, FileNotFoundError):
                 raise # Re-raise if it's not the expected connection error
            logger.warning(f"Decode worker {worker_id} connection failed (attempt {attempt + 1}/{setup_attempts}): {e}. Retrying...")
            time.sleep(0.2 * (attempt + 1)) # Exponential backoff
        except Exception as e:
            logger.error(f"Decode worker {worker_id}: Error during setup: {e}", exc_info=True)
            # Try to signal readiness even on error, but might fail if ready_queue connection is lost
            try:
                ready_queue.put(worker_id)
            except Exception as signal_e:
                logger.error(f"Decode worker {worker_id}: Failed to signal readiness after setup error: {signal_e}")
            if callback_runner:
                callback_runner.trigger_event(EventTypes.ON_WORKER_ERROR, worker_id=worker_id, worker_type="decode", error=e)
            return # Exit if setup fails due to other reasons

    if not setup_success:
        logger.error(f"Decode worker {worker_id} failed to connect after {setup_attempts} attempts.")
        # Try to signal readiness even on error, but might fail if ready_queue connection is lost
        try:
            ready_queue.put(worker_id)
        except Exception as signal_e:
            logger.error(f"Decode worker {worker_id}: Failed to signal readiness after setup error: {signal_e}")
        return

    while True:
        try:
            # Get data tuple: (response_queue_id, uid, timestamp, x_enc)
            response_queue_id, uid, timestamp, x_enc = input_queue.get(timeout=1.0)

            try:
                # Run before_decode callback if available
                if callback_runner:
                    callback_runner.run_callbacks('on_before_decode', lit_api)
                
                x = lit_api.decode_request(x_enc)
                
                # Run after_decode callback if available
                if callback_runner:
                    callback_runner.run_callbacks('on_after_decode', lit_api)
                # Put decoded data tuple: (response_queue_id, uid, timestamp, x)
                output_queue.put((response_queue_id, uid, timestamp, x))
            except Exception as e:
                logger.error(f"Decode worker {worker_id} failed processing request {uid}: {e}", exc_info=True)
                # TODO: How to signal error back to the client? Maybe put an error marker?
                # For now, just logging and skipping.
                pass

        except queue.Empty:
            # Normal timeout, continue loop
            continue
        except (RemoteError, EOFError, BrokenPipeError) as e:
            logger.warning(f"Decode worker {worker_id} lost connection to ready_queue: {e}. Exiting loop.")
            break # Exit loop if ready_queue connection is lost
        except (KeyboardInterrupt, SystemExit):
            logger.info(f"Decode worker {worker_id} received exit signal.")
            break
        except Exception as e:
            # Catch unexpected errors in the loop itself
            logger.exception(f"Decode worker {worker_id} encountered unexpected error: {e}")
            time.sleep(1) # Avoid tight loop on persistent error

    logger.info(f"Decode worker {worker_id} stopped.")


def preprocess_loop(
    api_cls: Type[LitAPI],
    lit_spec: Optional[LitSpec],
    input_queue: Any,  # Transport Queue (decoded_queue)
    output_queue: Any, # Transport Queue (preprocess_queue)
    worker_id: int,
    ready_queue: Any,
    callback_runner=None,
):
    """Worker loop dedicated to preprocessing data."""
    # Instantiate the API object
    lit_api = api_cls()
    logger.info(f"Preprocess worker {worker_id} starting.")
    # Initial setup and connection retry
    setup_attempts = 5
    setup_success = False
    for attempt in range(setup_attempts):
        try:
            logger.info(f"Preprocess worker {worker_id}: Initializing API...")
            lit_api.setup(device="cpu")
            logger.info(f"Preprocess worker {worker_id}: API setup complete.")
            ready_queue.put(worker_id) # Signal readiness via ready_queue
            setup_success = True
            break
        except (FileNotFoundError, AttributeError) as e:
            if "'ForkAwareLocal' object has no attribute 'connection'" not in str(e) and not isinstance(e, FileNotFoundError):
                 raise
            logger.warning(f"Preprocess worker {worker_id} connection failed (attempt {attempt + 1}/{setup_attempts}): {e}. Retrying...")
            time.sleep(0.2 * (attempt + 1))
        except Exception as e:
            logger.error(f"Preprocess worker {worker_id}: Error during setup: {e}", exc_info=True)
            # Try to signal readiness even on error, but might fail if ready_queue connection is lost
            try:
                ready_queue.put(worker_id)
            except Exception as signal_e:
                logger.error(f"Preprocess worker {worker_id}: Failed to signal readiness after setup error: {signal_e}")
            if callback_runner:
                callback_runner.trigger_event(EventTypes.ON_WORKER_ERROR, worker_id=worker_id, worker_type="preprocess", error=e)
            return

    if not setup_success:
        logger.error(f"Preprocess worker {worker_id} failed to connect after {setup_attempts} attempts.")
        # Try to signal readiness even on error, but might fail if ready_queue connection is lost
        try:
            ready_queue.put(worker_id)
        except Exception as signal_e:
            logger.error(f"Preprocess worker {worker_id}: Failed to signal readiness after setup error: {signal_e}")
        return

    while True:
        try:
            # Get decoded data tuple: (response_queue_id, uid, timestamp, x)
            response_queue_id, uid, timestamp, x = input_queue.get(timeout=1.0)

            try:
                # Run before_preprocess callback if available
                if callback_runner:
                    callback_runner.run_callbacks('on_before_preprocess', lit_api)
                
                x_processed = lit_api.preprocess(x)
                
                # Run after_preprocess callback if available
                if callback_runner:
                    callback_runner.run_callbacks('on_after_preprocess', lit_api)
                # Put preprocessed data tuple: (response_queue_id, uid, timestamp, x_processed)
                output_queue.put((response_queue_id, uid, timestamp, x_processed))
            except Exception as e:
                logger.error(f"Preprocess worker {worker_id} failed processing request {uid}: {e}", exc_info=True)
                # TODO: How to signal error back to the client?
                pass

        except queue.Empty:
            continue
        except (RemoteError, EOFError, BrokenPipeError) as e:
            logger.warning(f"Preprocess worker {worker_id} lost connection to ready_queue: {e}. Exiting loop.")
            break # Exit loop if ready_queue connection is lost
        except (KeyboardInterrupt, SystemExit):
            logger.info(f"Preprocess worker {worker_id} received exit signal.")
            break
        except Exception as e:
            logger.exception(f"Preprocess worker {worker_id} encountered unexpected error: {e}")
            time.sleep(1)

    logger.info(f"Preprocess worker {worker_id} stopped.")


def _process_predict_batch(
    batch, batch_inp_data, lit_api, worker_id, output_queue, max_batch_size, requires_batch, requires_unbatch,
    callback_runner=None
):
    """Helper function to process a collected batch for prediction."""
    try:
        # --- Prepare Batch --- 
        if requires_batch:
            batched_input = lit_api.batch(batch_inp_data)
        else:
            # If batch size is 1, or no batch method, pass single item
            if max_batch_size == 1:
                batched_input = batch_inp_data[0]
            else:
                 # Default batching (list of inputs) - model predict needs to handle this
                 batched_input = batch_inp_data

        # --- Predict --- 
        predict_start_time = time.monotonic()
        try:
            # Run before_predict callback if available
            if callback_runner:
                callback_runner.run_callbacks('on_before_predict', lit_api)
            
            results = lit_api.predict(batched_input)
            
            # Run after_predict callback if available
            if callback_runner:
                callback_runner.run_callbacks('on_after_predict', lit_api)
        except Exception as e:
            logger.error(f"Predict worker {worker_id} failed processing batch: {e}", exc_info=True)
            # Send error for each item in the failed batch
            # TODO: Define a standard error format/marker?
            for resp_id, req_uid, req_ts in batch:
                try:
                    # Send an error marker or exception back
                    error_payload = e # Or serialize it better
                    output_queue.put((resp_id, req_uid, req_ts, error_payload))
                except Exception as send_e:
                    logger.error(f"Predict worker {worker_id} failed to send error for {req_uid}: {send_e}")
            return
        
        predict_duration = time.monotonic() - predict_start_time
        logger.debug(f"Predict worker {worker_id} predict duration: {predict_duration:.4f}s for batch size {len(batch)}")

        # --- Unbatch Results --- 
        if requires_unbatch:
            outputs = lit_api.unbatch(results)
        else:
             # If batch size is 1, or no unbatch method, assume result is single item or list
            if max_batch_size == 1:
                outputs = [results] # Ensure it's a list
            else:
                # Assume results is already a list of outputs matching the input batch size
                if not isinstance(results, (list, tuple)) or len(results) != len(batch):
                    logger.warning(
                        f"Predict worker {worker_id}: Predict output count ({len(results) if isinstance(results, (list, tuple)) else 'N/A'}) does not match batch size ({len(batch)}). "
                        f"Ensure predict returns a list/tuple of results when max_batch_size > 1 or implement unbatch."
                    )
                    # Attempt to proceed if it's iterable, otherwise error
                    if isinstance(results, (list, tuple)):
                        outputs = results
                    else:
                        raise TypeError(f"Expected predict output to be a list/tuple for batch size > 1, got {type(results)}")
                else:
                    outputs = results

        # --- Send results --- 
        if len(outputs) != len(batch):
            logger.error(
                f"Predict worker {worker_id}: Unbatched output count ({len(outputs)}) does not match batch size ({len(batch)}). "
                f"Discarding batch."
            )
            # TODO: Send error markers back?
        else:
            for i, (resp_id, req_uid, req_ts) in enumerate(batch):
                output_queue.put((resp_id, req_uid, req_ts, outputs[i]))

    except Exception as e:
        logger.error(f"Predict worker {worker_id} failed processing batch: {e}", exc_info=True)
        # Send error for each item in the failed batch
        # TODO: Define a standard error format/marker?
        for resp_id, req_uid, req_ts in batch:
            try:
                # Send an error marker or exception back
                error_payload = e # Or serialize it better
                output_queue.put((resp_id, req_uid, req_ts, error_payload))
            except Exception as send_e:
                logger.error(f"Predict worker {worker_id} failed to send error for {req_uid}: {send_e}")


def predict_loop(
    api_cls: Type[LitAPI],
    lit_spec: Optional[LitSpec],
    device: str,
    input_queue: Any,  # Transport Queue (predict_input_queue)
    output_queue: Any, # Transport Queue (predict_output_queue)
    worker_id: int,
    max_batch_size: int,
    batch_timeout: float,
    ready_queue: Any,
    callback_runner=None,
):
    """Worker loop dedicated to model prediction/inference with batching."""
    lit_api = api_cls() # Instantiate API
    logger.info(f"Predict worker {worker_id} starting on device {device}.")
    # Initial setup and connection retry
    setup_attempts = 5

    # 1. Connection Setup (outside main try)
    setup_success = False
    for attempt in range(setup_attempts):
        try:
            # Call the main setup for predict workers
            logger.info(f"Predict worker {worker_id} on device {device}: Initializing API...")
            lit_api.setup(device=device)
            if lit_spec:
                lit_spec.setup(lit_api) # Allow spec to modify/setup API

            # Check if input queue is accessible (removed potentially problematic qsize)
            # If direct access fails early, it indicates a ready_queue issue.
            # A simple poll might be less problematic than qsize
            try:
                input_queue.empty() # Use empty() as a less problematic check
            except Exception as q_err:
                logger.warning(f"Predict worker {worker_id} setup: Input queue check failed: {q_err}")
                # Depending on the error, might indicate a deeper ready_queue issue

            logger.info(f"Predict worker {worker_id} on device {device}: API setup complete.")
            ready_queue.put(worker_id) # Signal readiness via ready_queue
            setup_success = True
            break # Exit retry loop on success
        except (FileNotFoundError, AttributeError, ConnectionRefusedError, BrokenPipeError) as e:
            if "'ForkAwareLocal' object has no attribute 'connection'" not in str(e) and not isinstance(e, FileNotFoundError):
                raise # Re-raise if it's not the expected connection error
            logger.warning(f"Predict worker {worker_id} connection attempt {attempt + 1}/{setup_attempts} failed: {e}")
            if attempt < setup_attempts - 1:
                time.sleep(0.2 * (attempt + 1)) # Exponential backoff
            else:
                logger.error(f"Predict worker {worker_id} failed to connect after {setup_attempts} attempts.")
                # Signal readiness (or failure) outside the main try/except to ensure it happens
                try:
                    ready_queue.put(worker_id)
                except Exception as signal_e:
                    logger.error(f"Predict worker {worker_id}: Failed to signal readiness after setup attempt: {signal_e}")
                return # Exit if connection setup fails completely
        except Exception as e:
            logger.exception(f"Predict worker {worker_id} encountered unexpected error during setup: {e}")
            # Signal readiness (or failure) outside the main try/except to ensure it happens
            try:
                ready_queue.put(worker_id)
            except Exception as signal_e:
                logger.error(f"Predict worker {worker_id}: Failed to signal readiness after setup attempt: {signal_e}")
            return # Exit if setup fails due to other reasons

    if not setup_success:
         # Should have returned above, but safeguard
         logger.error(f"Predict worker {worker_id} exiting due to connection failure.")
         return

    try: # Main try for API setup and processing loop
        # 2. API Setup (inside main try)
        if _is_torch_available():
            import torch
            torch.cuda.empty_cache()
        logger.info(f"Predict worker {worker_id} completed LitAPI setup on {device}.")
        # ready_event.set() # Moved to setup completion

        # 3. Main Loop (inside main try)
        exit_loop = False
        while True:
            if exit_loop:
                break # Exit if flag was set in previous iteration

            batch = []
            start_time = time.monotonic()
            timeout = batch_timeout

            # Collect batch
            while len(batch) < max_batch_size and timeout >= 0:
                try:
                    # Fetch data with calculated timeout
                    response_queue_id, uid, timestamp, data = input_queue.get(timeout=max(0.0, timeout))
                    batch.append((response_queue_id, uid, timestamp, data))
                    # Update timeout
                    timeout = batch_timeout - (time.monotonic() - start_time)
                except queue.Empty:
                    # Timeout occurred, break collection loop
                    break
                except (RemoteError, EOFError, BrokenPipeError) as e:
                    logger.warning(f"Predict worker {worker_id} lost connection to ready_queue during batch collection: {e}. Exiting loop.")
                    exit_loop = True # Set flag to exit outer loop
                    break # Break collection loop
                except Exception as e:
                    logger.error(f"Predict worker {worker_id} encountered error during batch collection: {e}", exc_info=True)
                    # Decide how to handle partial batches or errors; for now, break collection
                    break # Break collection, process any collected items

            if exit_loop:
                break # Exit if flag was set during collection

            # Process batch if not empty
            if batch:
                try:
                    inputs = [item[3] for item in batch]
                    # Perform prediction
                    _process_predict_batch(batch, inputs, lit_api, worker_id, output_queue, max_batch_size, lit_api.requires_batch, lit_api.requires_unbatch, callback_runner)
                except (RemoteError, EOFError, BrokenPipeError) as e:
                     logger.warning(f"Predict worker {worker_id} lost connection to ready_queue during batch processing: {e}. Exiting loop.")
                     exit_loop = True # Set flag to exit outer loop
                     # No break needed here, flag checked at start/end of outer loop
                except Exception as e:
                    # Log prediction errors
                    logger.error(f"Predict worker {worker_id} encountered error during prediction: {e}", exc_info=True)
                    # Handle prediction errors, e.g., send error response for affected requests
                    pass # Continue to next batch cycle
            # If no batch was collected and no timeout, sleep briefly to avoid busy-waiting
            elif timeout >= 0:
                 time.sleep(0.001)

    # Catches errors from API Setup or the main loop
    except (RemoteError, EOFError, BrokenPipeError) as e:
         logger.warning(f"Predict worker {worker_id} connection error during operation: {e}. Exiting.")
         # Signal readiness (or failure) outside the main try/except to ensure it happens
         try:
             ready_queue.put(worker_id)
         except Exception as signal_e:
             logger.error(f"Predict worker {worker_id}: Failed to signal readiness after setup attempt: {signal_e}")
    except Exception as e:
        # Log errors during setup or catastrophic failures
        logger.error(f"Predict worker {worker_id} failed during setup or loop: {e}", exc_info=True)
        # Signal readiness (or failure) outside the main try/except to ensure it happens
        try:
            ready_queue.put(worker_id)
        except Exception as signal_e:
            logger.error(f"Predict worker {worker_id}: Failed to signal readiness after setup attempt: {signal_e}")
    finally:
        logger.info(f"Predict worker {worker_id} exiting.")


def encode_loop(
    api_cls: Type[LitAPI],
    lit_spec: Optional[LitSpec],
    response_input_queue: Any,    # Transport Queue (predict_output_queue)
    result_queue: Any, # Main response transport object
    worker_id: int,
    ready_queue: Any,
    callback_runner=None,
):
    """Worker loop that encodes model predictions before sending back to the client."""
    # Instantiate the API object
    lit_api = api_cls() # Instantiate API
    logger.info(f"Encode worker {worker_id} starting.")
    # Initial setup and connection retry
    setup_attempts = 5
    setup_success = False
    for attempt in range(setup_attempts):
        try:
            # Minimal setup, if any, needed for encoding? Assume none for now.
            logger.info(f"Encode worker {worker_id}: Initializing API...")
            lit_api.setup(device="cpu")  # Encode always runs on CPU
            logger.info(f"Encode worker {worker_id}: API setup complete.")
            ready_queue.put(worker_id) # Signal readiness via ready_queue
            setup_success = True
            break
        except (FileNotFoundError, AttributeError) as e:
            if "'ForkAwareLocal' object has no attribute 'connection'" not in str(e) and not isinstance(e, FileNotFoundError):
                 raise
            logger.warning(f"Encode worker {worker_id} connection failed (attempt {attempt + 1}/{setup_attempts}): {e}. Retrying...")
            time.sleep(0.2 * (attempt + 1))
        except Exception as e:
            logger.error(f"Encode worker {worker_id}: Error during setup: {e}", exc_info=True)
            # Try to signal readiness even on error, but might fail if ready_queue connection is lost
            try:
                ready_queue.put(worker_id)
            except Exception as signal_e:
                logger.error(f"Encode worker {worker_id}: Failed to signal readiness after setup error: {signal_e}")
            if callback_runner:
                callback_runner.trigger_event(EventTypes.ON_WORKER_ERROR, worker_id=worker_id, worker_type="encode", error=e)
            return

    if not setup_success:
        logger.error(f"Encode worker {worker_id} failed to connect after {setup_attempts} attempts.")
        # Try to signal readiness even on error, but might fail if ready_queue connection is lost
        try:
            ready_queue.put(worker_id)
        except Exception as signal_e:
            logger.error(f"Encode worker {worker_id}: Failed to signal readiness after setup error: {signal_e}")
        return

    logger.info(f"Encode worker {worker_id} entering main loop.")
    while True:
        try:
            # Get prediction result tuple: (response_queue_id, uid, timestamp, result)
            # Result could be the actual prediction or an Exception from predict_loop
            logger.debug(f"Encode worker {worker_id}: Attempting get() from queue proxy: {response_input_queue}") # Added log
            response_queue_id, uid, timestamp, result = response_input_queue.get(timeout=1.0)

            try:
                # Check if the result from predict_loop was an error
                if isinstance(result, Exception):
                    # TODO: Implement proper error serialization/handling
                    # For now, just pass the exception object, which might not be ideal
                    # Need a way to signal an error to the client via encode_response
                    # or by setting a specific status code/payload.
                    encoded_response = lit_api.encode_response(result) # Let encode_response handle the error object
                    logger.warning(f"Encode worker {worker_id} processing error for request {uid}: {result}")
                else:
                    encoded_response = lit_api.encode_response(result)
                
                # Send final tuple: (response_queue_id, uid, timestamp, encoded_response)
                result_queue.put((response_queue_id, uid, timestamp, encoded_response))

            except Exception as e:
                logger.error(f"Encode worker {worker_id} failed processing request {uid}: {e}", exc_info=True)
                # Try sending an error back if encoding failed
                try:
                    error_payload = f"Encoding failed: {e}" # Simple error message
                    result_queue.put((response_queue_id, uid, timestamp, error_payload))
                except Exception as send_e:
                    logger.error(f"Encode worker {worker_id} failed to send error for {uid}: {send_e}")

        except queue.Empty:
            # Normal timeout, continue loop
            continue
        except (BrokenPipeError, EOFError, ConnectionResetError, OSError) as e:
            # Specific ready_queue connection errors
            logger.warning(f"Encode worker {worker_id} lost connection to ready_queue: {e}. Exiting loop.")
            break
        except Exception as e:
            # General errors during the loop
            logger.exception(f"Encode worker {worker_id} encountered error in main loop: {e}")
            # We might want to break here depending on the error, or try to continue
            # For now, let's break on general exceptions too, as they might indicate state corruption
            break

    logger.info(f"Encode worker {worker_id} stopped.")
