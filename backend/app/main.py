"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import game_rosters, games, players, timeline, video_upload, videos

app = FastAPI(
    title="Basketball Video Analyzer API",
    description="API for analyzing youth basketball game videos with player tracking and play tagging",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(games.router, prefix="/api")
app.include_router(players.router, prefix="/api")
app.include_router(videos.router, prefix="/api")
app.include_router(game_rosters.router, prefix="/api")
app.include_router(video_upload.router, prefix="/api")
app.include_router(timeline.router, prefix="/api")


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint returning API information."""
    return {
        "name": "Basketball Video Analyzer API",
        "version": __version__,
        "status": "running",
    }


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
