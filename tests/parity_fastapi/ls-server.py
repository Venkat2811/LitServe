import argparse
import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor

import PIL
import torch
import torchvision

import litserve as ls

# Force CPU for testing to avoid CUDA issues
device = "cpu"

# Configuration for different devices
conf = {
    "cpu": {"batch_size": 4, "workers_per_device": 1},  # Reduced for testing
}

# Set float32 matrix multiplication precision if GPU is available and capable
if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
    torch.set_float32_matmul_precision("high")


class ImageClassifierAPI(ls.LitAPI):
    def setup(self, device):
        print(f"Setting up ImageClassifierAPI on device: {device}")
        try:
            # Initialize model with error handling
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            self.image_processing = weights.transforms()
            
            # Create a smaller model for testing to reduce memory usage
            # Handle device which might be a string or a list
            device_str = device[0] if isinstance(device, list) else device
            
            if 'cuda' in str(device_str):
                # For CUDA devices, use a smaller model
                print(f"Initializing ResNet18 on {device_str}")
                # Extract device index if present
                if ':' in str(device_str):
                    device_idx = int(device_str.split(':')[1])
                    torch.cuda.set_device(device_idx)
                else:
                    torch.cuda.set_device(0)
                self.model = torchvision.models.resnet18(weights=None).eval().to(device_str)
            else:
                # For CPU, use the regular model
                self.model = torchvision.models.resnet18(weights=None).eval().to(device_str)
                
            self.pool = ThreadPoolExecutor(max(1, os.cpu_count() // 2))  # Use fewer threads
            self.device = device
            print(f"Successfully initialized model on {device}")
        except Exception as e:
            print(f"Error initializing model on {device}: {e}")
            # Re-raise to properly report the error
            raise

    def decode_request(self, request):
        return request["image_data"]

    def batch(self, image_data_list):
        def process_image(image_data):
            image = base64.b64decode(image_data)
            pil_image = PIL.Image.open(io.BytesIO(image)).convert("RGB")
            return self.image_processing(pil_image)

        inputs = list(self.pool.map(process_image, image_data_list))
        return torch.stack(inputs).to(self.device)

    def predict(self, x):
        with torch.inference_mode():
            outputs = self.model(x)
            _, predictions = torch.max(outputs, 1)
        return predictions

    def unbatch(self, outputs):
        return outputs.tolist()

    def encode_response(self, output):
        return {"output": output}


def main(batch_size: int, workers_per_device: int, execution_mode: str = "default", port: int = 8000):
    print(f"Starting LitServe server with execution_mode={execution_mode}, batch_size={batch_size}, workers={workers_per_device}")
    api = ImageClassifierAPI()
    
    # Use even simpler configuration for full_parallel mode
    if execution_mode == "full_parallel":
        print("Using simplified configuration for full_parallel mode")
        batch_size = 1  # Disable batching for full_parallel mode
        workers_per_device = 1  # Use single worker per device
    
    server = ls.LitServer(
        api,
        accelerator="cpu",  # Force CPU for testing
        devices=1,
        max_batch_size=batch_size,
        batch_timeout=0.0,  # No batch timeout
        timeout=30,  # Longer timeout
        workers_per_device=workers_per_device,
        execution_mode=execution_mode,
        fast_queue=False,  # Disable ZMQ for simpler testing
    )
    
    # Print server configuration
    print(f"Server configuration:")
    print(f"  Execution mode: {execution_mode}")
    print(f"  Accelerator: cpu")
    print(f"  Max batch size: {batch_size}")
    print(f"  Workers per device: {workers_per_device}")
    
    server.run(port=port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LitServe server with different execution modes")
    parser.add_argument(
        "--execution-mode", 
        choices=["default", "full_parallel"], 
        default="default",
        help="Execution mode for LitServe (default or full_parallel)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to run the server on"
    )
    args = parser.parse_args()
    
    # Get device-specific configuration and add command line arguments
    config = conf[device].copy()
    config["execution_mode"] = args.execution_mode
    config["port"] = args.port
    
    main(**config)
