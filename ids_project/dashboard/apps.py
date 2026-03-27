from django.apps import AppConfig


class DashboardConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'dashboard'

    def ready(self):
        # Register signal handlers (post_save hooks for websocket broadcasts).
        from . import traffic_processor  # noqa: F401
