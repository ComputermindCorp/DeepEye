from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
import classification.routing
from django.urls import path
import sys
application = ProtocolTypeRouter({
# (http->django views is added by default)
'websocket': AuthMiddlewareStack(
    URLRouter([
        path(r'ws/classification', classification.consumers.Classification.as_asgi())
        ])
    ),
})
