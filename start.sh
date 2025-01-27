#!/bin/bash
echo Starting NGinx
service nginx start

echo Starting uWSGI

uwsgi --ini uwsgi.ini
