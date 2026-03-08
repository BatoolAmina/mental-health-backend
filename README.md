# Mental Health Backend

This folder contains the Flask-based backend for the Mental Health web
application.  The service exposes authentication endpoints and prediction
routes, and stores chat history in MongoDB.

## Environment Variables

The backend uses [`python-dotenv`](https://pypi.org/project/python-dotenv/)
to load environment variables from a `.env` file in the backend directory.

Required variables:

- `SECRET_KEY` – JWT secret used to sign tokens.
- `MONGO_URI` – connection string for MongoDB.  For local development, you
  can use:
  ```
  MONGO_URI=mongodb://127.0.0.1:27017/mental_health_db
  ```
  If you use a MongoDB Atlas cluster, ensure your machine can reach the
  DNS records or supply a non-`+srv` URI.  The application will automatically
  fall back to `mongodb://127.0.0.1:27017` if the primary URI cannot be
  resolved (useful when working offline).

Optional variables:

- `GOOGLE_CLIENT_ID` – for Google sign‑in support.
- `DEBUG` – set to any value to enable verbose logging from the backend.

## Running the server

Activate your virtual environment and install dependencies from
`requirements.txt`:

```powershell
venv\Scripts\activate
pip install -r requirements.txt
```

Then start the app:

```powershell
python app.py
```

You should see log messages about MongoDB connectivity; if the service
fails to connect, the process will exit with a descriptive error.

---

See the frontend README for instructions about the Next.js client.
