#!/usr/bin/env python
# -*- coding: utf-8 -*-

from werkzeug.serving import run_simple
from myproject import make_app

app = make_app(...)
run_simple('localhost', 5000, app,
           ssl_context=('ssl_.crt',
                        'ssl_.key'))