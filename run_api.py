from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import uvicorn


def main():
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()