#!/bin/bash

source /media/pleiades/vault/projects/202110-nlos-render/mitsuba3-transient/mitsuba3/build/setpath.sh
cd /media/pleiades/vault/projects/202110-nlos-render/mitsuba3-transient/mitsuba3-transient
python3 -m pip install .
cd /media/pleiades/vault/projects/202110-nlos-render/mitsuba3-transient/mitsuba3-transient/test/z-scene-properties
python3 scene_properties.py