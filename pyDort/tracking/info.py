from .tracklet import (
    TrackletPedestrianCV,
    TrackletVehicleAdaptiveCTRV,
    TrackletVehicleCTRV,
    TrackletVehicleCV,
)

tracklets = {
    'pedestrian-cv': TrackletPedestrianCV,
    'pedestrian-cv-v0': TrackletVehicleCV,
    'vehicle-cv': TrackletVehicleCV,
    'vehicle-ctrv': TrackletVehicleCTRV,
    'vehicle-adaptive-ctrv': TrackletVehicleAdaptiveCTRV
    }
