#!/bin/bash
MJSON_PATH=${1}

if ! gzip -t ${MJSON_PATH} >/dev/null 2>&1; then
  cp ${MJSON_PATH} ${MJSON_PATH%.mjlog}
  mv ${MJSON_PATH} ${MJSON_PATH}.bak
  gzip ${MJSON_PATH%.mjlog} -S .mjlog
fi
OUTPUT_PATH=${2}
mjai convert ${MJSON_PATH} ${OUTPUT_PATH}