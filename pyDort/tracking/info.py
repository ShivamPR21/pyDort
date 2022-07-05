from .tracklet import (
    TrackletPedestrianCV,
    TrackletVehicleAdaptiveCTRV,
    TrackletVehicleCTRV,
    TrackletVehicleCV,
)

tracklets = {
    'pedestrian-cv': TrackletPedestrianCV,
    'vehicle-cv': TrackletVehicleCV,
    'vehicle-ctrv': TrackletVehicleCTRV,
    'vehicle-adaptive-ctrv': TrackletVehicleAdaptiveCTRV
    }
