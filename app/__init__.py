# app/__init__.py

from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Import and register routes
    from .routes import routes as routes_blueprint
    app.register_blueprint(routes_blueprint)
    
    return app
