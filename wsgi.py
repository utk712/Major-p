from app import app

# Alias for WSGI application servers (Render default: gunicorn wsgi:application)
application = app

if __name__ == "__main__":
    app.run()
