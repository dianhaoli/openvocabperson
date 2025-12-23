#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Script for Hierarchical Vision Analysis Pipeline

This script demonstrates both synchronous and async streaming modes
of the HierarchicalPipeline.

Usage:
    python demo.py [image_path]
    python demo.py                    # Uses default test image
    python demo.py my_image.jpg       # Uses specified image
"""

import asyncio
import sys
import time

from pipeline import HierarchicalPipeline, PipelineConfig, create_pipeline, ANALYSIS_PROMPT


def demo_sync(image_path: str):
    """Demonstrate synchronous analysis mode."""
    print("\n" + "=" * 70)
    print("🔄 SYNCHRONOUS MODE DEMO")
    print("=" * 70)
    
    # Create pipeline with accuracy-focused settings
    config = PipelineConfig(
        use_quantization=True,      # INT8 for good accuracy + speed
        use_torch_compile=False,    # Compile can be tricky with quantization
        max_new_tokens=35,          # Concise outputs
        batch_size=4,               # Batch VLM inference
    )
    pipeline = HierarchicalPipeline(config)
    pipeline._load_models()
    
    # Time the analysis
    start_time = time.perf_counter()
    results = pipeline.analyze(image_path)
    elapsed = time.perf_counter() - start_time
    
    # Print results
    pipeline.print_results(results)
    
    print(f"\n⏱️  Total analysis time: {elapsed:.2f}s")
    print(f"   Crops analyzed: {len(results)}")
    print(f"   Avg per crop: {elapsed / max(len(results), 1):.2f}s")
    
    # Visualize
    try:
        pipeline.visualize_results(results)
    except Exception as e:
        print(f"   (Visualization skipped: {e})")
    
    return pipeline, results


async def demo_async_streaming(image_path: str):
    """Demonstrate async streaming analysis mode."""
    print("\n" + "=" * 70)
    print("⚡ ASYNC STREAMING MODE DEMO")
    print("=" * 70)
    
    # Create pipeline
    pipeline = create_pipeline(use_quantization=True, use_compile=False)
    await pipeline.initialize()
    
    start_time = time.perf_counter()
    results_collected = []
    
    print("\n📡 Streaming results as they arrive:\n")
    
    async for event in pipeline.analyze_streaming(image_path):
        elapsed = time.perf_counter() - start_time
        
        if event.event_type == "detection_complete":
            print(f"[{elapsed:5.2f}s] 🔍 Detection complete: {event.data['num_detections']} objects found")
            for det in event.data['detections'][:5]:  # Show first 5
                print(f"         • {det['class']} (conf: {det['confidence']:.2f})")
            if len(event.data['detections']) > 5:
                print(f"         ... and {len(event.data['detections']) - 5} more")
        
        elif event.event_type == "routing_complete":
            print(f"[{elapsed:5.2f}s] 🔀 Routing complete:")
            print(f"         • Skip VLM: {event.data['yolo_only']}")
            print(f"         • Low conf: {event.data['low_confidence']}")
            print(f"         • Full VLM: {event.data['vlm_full']}")
        
        elif event.event_type == "crop_analyzed":
            data = event.data
            stage = data.get('stage', 'unknown')
            idx = data.get('index', '?')
            analysis = data.get('analysis')  # This is the analysis text
            
            if analysis:
                # VLM result - show preview
                preview = analysis[:60].replace('\n', ' ') + "..." if len(analysis) > 60 else analysis.replace('\n', ' ')
                print(f"[{elapsed:5.2f}s] ✅ Crop {idx} ({stage}): {preview}")
            else:
                # Skipped
                print(f"[{elapsed:5.2f}s] ⏭️  Crop {idx} ({stage}): {data.get('reason', 'skipped')}")
            
            results_collected.append(data)
        
        elif event.event_type == "complete":
            print(f"\n[{elapsed:5.2f}s] 🏁 Analysis complete!")
    
    total_time = time.perf_counter() - start_time
    print(f"\n⏱️  Total streaming time: {total_time:.2f}s")
    print(f"   Results received: {len(results_collected)}")
    
    return pipeline, results_collected


def demo_custom_prompt(pipeline: HierarchicalPipeline, image_path: str):
    """Demonstrate using a custom prompt."""
    print("\n" + "=" * 70)
    print("📝 CUSTOM PROMPT DEMO")
    print("=" * 70)
    
    custom_prompt = """What is directly visible?
1. Actions happening now
2. Objects being held
3. Clothing visible
If unsure, say 'unclear'."""
    
    print(f"\nCustom prompt:\n{custom_prompt}\n")
    
    results = pipeline.analyze(image_path, prompt=custom_prompt)
    
    # Show just VLM results
    vlm_results = [r for r in results if r.analysis_text]
    for result in vlm_results[:3]:  # Show first 3
        print(f"Crop {result.crop_info.detection.index}:")
        print(f"  {result.analysis_text}")
        print()


def benchmark_comparison(image_path: str):
    """Compare different optimization settings."""
    print("\n" + "=" * 70)
    print("📊 OPTIMIZATION BENCHMARK")
    print("=" * 70)
    
    configs = [
        ("No optimizations", PipelineConfig(use_quantization=False, use_torch_compile=False)),
        ("INT8 quantization", PipelineConfig(use_quantization=True, quantization_bits=8, use_torch_compile=False)),
        ("torch.compile only", PipelineConfig(use_quantization=False, use_torch_compile=True)),
    ]
    
    results = []
    
    for name, config in configs:
        print(f"\n🔧 Testing: {name}")
        
        pipeline = HierarchicalPipeline(config)
        
        try:
            pipeline._load_models()
            
            # Warmup
            _ = pipeline.analyze(image_path)
            
            # Timed run
            start = time.perf_counter()
            analysis = pipeline.analyze(image_path)
            elapsed = time.perf_counter() - start
            
            vlm_count = len([r for r in analysis if r.analysis_text])
            results.append((name, elapsed, vlm_count))
            
            print(f"   Time: {elapsed:.2f}s for {vlm_count} VLM analyses")
            
            pipeline.cleanup()
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append((name, None, 0))
    
    # Summary table
    print("\n" + "─" * 50)
    print(f"{'Configuration':<25} {'Time':>10} {'VLM Crops':>10}")
    print("─" * 50)
    for name, elapsed, count in results:
        time_str = f"{elapsed:.2f}s" if elapsed else "failed"
        print(f"{name:<25} {time_str:>10} {count:>10}")
    print("─" * 50)


def main():
    """Main demo entry point."""
    # Get image path from command line or use default
    image_path = sys.argv[1] if len(sys.argv) > 1 else "YFQdcXrsu64BJMMhEl6k2WPjorzYG4.jpg"
    
    print("╔" + "═" * 68 + "╗")
    print("║" + " HIERARCHICAL VISION ANALYSIS PIPELINE - DEMO ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n📷 Image: {image_path}")
    
    # Run demos
    try:
        # 1. Synchronous mode
        pipeline, results = demo_sync(image_path)
        
        # 2. Custom prompt (reuse loaded pipeline)
        demo_custom_prompt(pipeline, image_path)
        
        # Cleanup
        pipeline.cleanup()
        
        # 3. Async streaming mode
        asyncio.run(demo_async_streaming(image_path))
        
        # 4. Benchmark (optional, takes longer)
        # Skip in non-interactive mode
        try:
            run_benchmark = input("\n🔧 Run optimization benchmark? (y/n): ").lower() == 'y'
            if run_benchmark:
                benchmark_comparison(image_path)
        except EOFError:
            print("\n(Skipping benchmark in non-interactive mode)")
        
    except FileNotFoundError:
        print(f"\n❌ Error: Image not found at '{image_path}'")
        print("   Please provide a valid image path as argument.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()

