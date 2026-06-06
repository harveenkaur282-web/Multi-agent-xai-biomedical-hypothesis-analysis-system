import sys
import traceback

print("Testing imports and pipeline build...")
try:
    from pipeline.graph import build_pcos_pipeline
    print("Successfully imported build_pcos_pipeline!")
    pipeline = build_pcos_pipeline()
    print("Successfully built PCOS pipeline!")
except Exception as e:
    print("FAILED to build PCOS pipeline:")
    traceback.print_exc()
