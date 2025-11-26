# Notes Backend - Environment and CORS

This backend expects configuration via environment variables loaded from `.env`.

Required:
- MONGODB_URL
- MONGODB_DB
- JWT_SECRET

Optional:
- JWT_ALG (default: HS256)
- ACCESS_TOKEN_EXPIRE_MINUTES (default: 60)
- CORS_ORIGINS (comma-separated; defaults to http://localhost:3000)

Example files are provided in `.env.example`. Copy to `.env` and edit as needed.

The backend exposes CORS with origins from `CORS_ORIGINS`. Ensure the frontend URL (e.g., http://localhost:3000) is included.
