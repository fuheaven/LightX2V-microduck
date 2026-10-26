# Packaged development recordings

These nine MP4 clips, posters and reports come from the project's actual MuJoCo
experiments. No learned weights, training data, credentials or live service URLs
are included. Total size is about 12 MB; a fresh clone can show the gallery offline.

`catalog.json` contains SHA256 and size for every media/evidence file. The service
verifies them and checks the clip outcome against its episode. The clear-room
score is 59/60 **development realtime** cases; one-obstacle is 27/60 **offline**;
random-obstacle is 2/21 **development realtime**. These cohorts must stay separate.
All recordings use the retained `clear-room-realtime-control-pilot-v1 / step 250`
model. The clear-room unoptimized comparison is labeled separately.

The left half of each recording is an observer view re-rendered from the original
recorded joint states; the right half contains original head-camera observations.
Videos play at simulation-time 1×; they are not streams of a currently running
world. New runs appear only in the Live Playground, never in this curated catalog.

Recordings were produced by this project; they depict the official MicroDuck
robot assets. The upstream asset licensing and attribution in
`docs/microduck/THIRD_PARTY.md` still apply. No MJCF/mesh bundle is redistributed
here. Raw report provenance is retained; historical source paths are provenance,
not deployment requirements. Formal acceptance and hardware validation are false.
