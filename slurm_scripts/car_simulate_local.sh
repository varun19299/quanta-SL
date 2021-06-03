# Start Job here
python pypbrt/simulate.py pbrt.scene=sportscar pbrt.filename=sportscar-area-lights

python pypbrt/simulate.py pbrt.scene=sportscar projector.index='range(0,21)' -m
