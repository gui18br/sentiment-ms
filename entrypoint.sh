#!/bin/sh

rm -rf /tmp/prometheus
mkdir -p /tmp/prometheus

exec "$@"