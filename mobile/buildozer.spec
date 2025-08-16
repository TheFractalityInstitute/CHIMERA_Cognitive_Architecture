# mobile/buildozer.spec
[app]
title = CHIMERA
package.name = chimera
package.domain = com.fractality

source.dir = .
source.include_exts = py,png,jpg,kv,atlas

version = 1.0.0

requirements = python3,kivy,numpy,websockets,aiohttp

[buildozer]
log_level = 2
