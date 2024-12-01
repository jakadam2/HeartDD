import torch

# from description.skeletonization import HDBifurcationPDF
# from description.layers import HDModel,HDHOGLayer
from description.characteristics import *

from typing import Union


class LesionDescriber:

    BOX_SIZE = 55

    def __init__(self):
        self.blunt_stump = BluntStumpClassifier(model_path = 'server/description/weights/blunt_stump_prob_model.pkl')
        self.heavy_calcification = HeavyCalcificationClassifier(model_path = 'server/description/weights/hc_prob_model.pkl')
        self.thrombus = ThrombusClassifier(model_path = 'server/description/weights/thrombus_prob_model.pkl')
        self.total_oclusion = TotalOclusionClassifier(model_path = 'server/description/weights/total_oclusion_prob_model.pkl')

    def __call__(self,image:torch.Tensor,mask:torch.Tensor,coords:Union[int,int]):
        return self.predict(image,mask,coords)   
    
    def predict(self,image:torch.Tensor,mask:torch.Tensor,coords:Union[int,int]) -> dict[str:float]:
        # bifurcation_prob = HDBifurcationPDF(mask.numpy())
        bifurcation_prob = BifurcationClassifier(mask)(*coords)
        turtosity_prob = SevereTortuosityClassifier(mask,5,0.3)()
        # croped = image[:,int(x - 0.5*LesionDescriber.BOX_SIZE):int(x + 0.5*LesionDescriber.BOX_SIZE), 
        #                int(y - 0.5*LesionDescriber.BOX_SIZE):int(y + 0.5*LesionDescriber.BOX_SIZE)].unsqueeze(0).to('cuda')
        
        # with torch.no_grad():
        #     prob = self.model(croped)

        # prob = prob.squeeze()
        
        return {
                'BIFURCATION':bifurcation_prob,
                'AORTO_OSTIAL_STENOSIS': 0,
                'BLUNT_STUMP':self.blunt_stump(image,coords),
                'TOTAL_OCLUSION':self.total_oclusion(image,coords),
                'HEAVY_CALCIFICATION':self.heavy_calcification(image,coords),
                'SEVERE_TORTUOSITY':turtosity_prob, 
                'THROMBUS':self.thrombus(image,coords),
            }

        