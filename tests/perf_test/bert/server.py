"""A BERT-Large text classification server with batching to be used for benchmarking."""

import os
import sys
import time
import torch
import multiprocessing as mp
from jsonargparse import CLI
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertConfig

import litserve as ls

# Set float32 matrix multiplication precision if GPU is available and capable
if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
    torch.set_float32_matmul_precision("high")

# set dtype to bfloat16 if CUDA is available
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32


class HuggingFaceLitAPI(ls.LitAPI):
    def __init__(self):
        super().__init__()
        # Flags to track initialization state
        self.model = None
        self.tokenizer = None
        self.device = None  # Store the assigned device

    def setup(self, device):
        """Setup the model and tokenizer in the worker process."""
        print(f"\n===> Setup received device: {device}")

        # Store the assigned device for use in predict method
        # Accept either a single device or a list (for compatibility)
        if isinstance(device, list):
            # In parallel modes, a list of devices might be passed to each predict worker.
            # We'll just use the first one for this simple example.
            self.device = device[0]
        else:
            # In default mode, a single device string is passed.
            self.device = device

        print(f"===> Worker using device: {self.device}")

        try:
            # Load model and tokenizer directly in the worker
            model_name = "google-bert/bert-large-uncased"
            print(f"===> Loading tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"===> Loading model: {model_name}")
            # Load model with config if specific config needed, otherwise directly
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=dtype)

            print(f"===> Moving model to device: {self.device}")
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            print(f"===> Setup completed successfully on device {self.device}")

        except Exception as e:
            print(f"Error in setup on device {self.device}: {e}")
            import traceback
            traceback.print_exc()
            # Re-raise the exception to signal setup failure to LitServer
            raise

    def decode_request(self, request: dict):
        """Extract text from the request."""
        return request["text"]

    def batch(self, inputs):
        """Batch the inputs."""
        try:
            if self.tokenizer is None:
                # This shouldn't happen if setup completed successfully
                print("Tokenizer not found in batch method, setup might have failed.")
                raise RuntimeError("Tokenizer is not initialized.")
            return self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        except Exception as e:
            print(f"Error in batch: {e}")
            # Re-raise or return an indicator of failure
            raise

    def predict(self, inputs):
        """Run prediction with error handling."""
        # The device is set during setup
        if self.model is None or self.device is None:
            print("Model or device not initialized in predict, setup might have failed.")
            raise RuntimeError("Model or device not initialized during predict.")

        try:
            # Ensure inputs are on the correct device
            processed_inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = self.model(**processed_inputs)
                logits = outputs.logits
                # Move result back to CPU for unbatching/further processing if needed
                return torch.argmax(logits, dim=1).cpu()

        except Exception as e:
            print(f"Error in predict on device {self.device}: {e}")
            import traceback
            traceback.print_exc()
            # Re-raise to let LitServe handle worker errors
            raise

    def unbatch(self, outputs):
        """Convert tensor outputs to Python lists."""
        try:
            return outputs.tolist()
        except Exception as e:
            print(f"Error in unbatch: {e}")
            # Return dummy output if conversion fails
            return [0] * len(outputs) if hasattr(outputs, '__len__') else [0]

    def encode_response(self, output):
        """Format the final response."""
        return {"label_idx": output}


def main(
    batch_size: int = 10,
    batch_timeout: float = 0.01,
    devices: int = 1,  # Changed from 2 to 1 for default mode compatibility
    workers_per_device=2,
    mode: str = "default",  # execution mode: 'default' or 'full_parallel'
):
    print(locals())

    # Create and initialize API - Model loading happens in worker setup
    api = HuggingFaceLitAPI()

    # Create the server with appropriate configuration
    server = ls.LitServer(
        api,
        max_batch_size=batch_size,
        workers_per_device=workers_per_device,
        accelerator="auto",
        devices=devices,
        batch_timeout=batch_timeout,
        timeout=200,
        fast_queue=True,  # Important for performance
        execution_mode=mode,
    )

    # Run the server
    try:
        server.run(log_level="warning", num_api_servers=4, generate_client_file=False)
    except Exception as e:
        print(f"Error in server.run: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    CLI(main)
