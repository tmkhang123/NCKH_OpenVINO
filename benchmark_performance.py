"""
Performance Benchmark Script
Test inference speed with different configurations
"""

import time
import numpy as np
from PIL import Image
import openvino as ov
import openvino_genai as ov_genai
from pathlib import Path

MODEL_PATH = "models/sd15_int8_ov"

def benchmark_config(device, steps, strength, guidance, resolution=(512, 512)):
    """Benchmark a specific configuration"""
    print(f"\n{'='*60}")
    print(f"Testing: Device={device}, Steps={steps}, Strength={strength}")
    print(f"Resolution: {resolution}, Guidance={guidance}")
    print(f"{'='*60}")

    try:
        # Load model
        print("Loading model...")
        load_start = time.time()
        pipe = ov_genai.Image2ImagePipeline(MODEL_PATH, device)
        load_time = time.time() - load_start
        print(f"✓ Model loaded in {load_time:.2f}s")

        # Create dummy image
        img = Image.new('RGB', resolution, color=(128, 128, 128))
        img_array = np.array(img, dtype=np.uint8)
        img_array = img_array.reshape(1, img_array.shape[0], img_array.shape[1], 3)
        tensor = ov.Tensor(img_array)

        # Warmup run
        print("Warmup run...")
        _ = pipe.generate(
            "a beautiful landscape",
            tensor,
            num_inference_steps=5,
            strength=0.5,
            guidance_scale=5.0,
            seed=42
        )

        # Actual benchmark (3 runs)
        times = []
        for i in range(3):
            print(f"Run {i+1}/3...", end=" ", flush=True)
            start = time.time()
            out = pipe.generate(
                "a beautiful landscape with mountains and rivers",
                tensor,
                negative_prompt="blurry, low quality",
                num_inference_steps=steps,
                strength=strength,
                guidance_scale=guidance,
                seed=42
            )
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"{elapsed:.2f}s")

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"\n📊 Results:")
        print(f"   Average: {avg_time:.2f}s ± {std_time:.2f}s")
        print(f"   Min: {min(times):.2f}s")
        print(f"   Max: {max(times):.2f}s")

        return {
            'device': device,
            'steps': steps,
            'strength': strength,
            'guidance': guidance,
            'resolution': resolution,
            'avg_time': avg_time,
            'std_time': std_time,
            'min_time': min(times),
            'max_time': max(times)
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("🚀 OpenVINO Image2Image Performance Benchmark")
    print("="*60)

    # Check available devices
    core = ov.Core()
    devices = core.available_devices
    print(f"Available devices: {devices}")

    # Test configurations
    configs = [
        # Original (slow)
        {'device': 'CPU', 'steps': 30, 'strength': 0.8, 'guidance': 7.5},

        # Optimized (fast)
        {'device': 'CPU', 'steps': 15, 'strength': 0.6, 'guidance': 5.0},
        {'device': 'CPU', 'steps': 12, 'strength': 0.6, 'guidance': 4.0},
        {'device': 'CPU', 'steps': 10, 'strength': 0.5, 'guidance': 4.0},

        # Resolution tests
        {'device': 'CPU', 'steps': 15, 'strength': 0.6, 'guidance': 5.0, 'resolution': (384, 384)},
    ]

    # Add GPU tests if available
    if 'GPU' in devices or 'GPU.0' in devices:
        gpu_device = 'GPU' if 'GPU' in devices else 'GPU.0'
        configs.extend([
            {'device': gpu_device, 'steps': 15, 'strength': 0.6, 'guidance': 5.0},
            {'device': gpu_device, 'steps': 20, 'strength': 0.7, 'guidance': 6.0},
        ])

    results = []
    for i, config in enumerate(configs):
        print(f"\n\n{'#'*60}")
        print(f"Configuration {i+1}/{len(configs)}")
        print(f"{'#'*60}")

        resolution = config.pop('resolution', (512, 512))
        result = benchmark_config(resolution=resolution, **config)
        if result:
            results.append(result)

    # Summary
    print(f"\n\n{'='*60}")
    print("📈 SUMMARY")
    print(f"{'='*60}")
    print(f"{'Device':<8} {'Steps':<6} {'Str':<5} {'Guid':<5} {'Res':<10} {'Time (s)':<10}")
    print("-" * 60)

    for r in results:
        res_str = f"{r['resolution'][0]}x{r['resolution'][1]}"
        print(f"{r['device']:<8} {r['steps']:<6} {r['strength']:<5.1f} "
              f"{r['guidance']:<5.1f} {res_str:<10} {r['avg_time']:<10.2f}")

    # Best config
    if results:
        fastest = min(results, key=lambda x: x['avg_time'])
        print(f"\n⚡ Fastest configuration:")
        print(f"   Device: {fastest['device']}")
        print(f"   Steps: {fastest['steps']}")
        print(f"   Strength: {fastest['strength']}")
        print(f"   Guidance: {fastest['guidance']}")
        print(f"   Resolution: {fastest['resolution']}")
        print(f"   Time: {fastest['avg_time']:.2f}s")

if __name__ == "__main__":
    main()
