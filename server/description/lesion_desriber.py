import torch

from skeletonization import HDBifurcationPDF
from layers import HDModel,HDHOGLayer

from typing import Union


class LesionDescriber:

    BOX_SIZE = 55

    def __init__(self):
        hog_layer = HDHOGLayer()
        self.model = HDModel(hog=hog_layer)

    def __call__(self,image:torch.Tensor,mask:torch.Tensor,coords:Union[int,int]):
        return self.predict(image,mask,coords)
    
    def predict_list(self,image:torch.Tensor,mask:torch.Tensor,coords:list[Union[int,int]]) -> list[Union[str,float]]:
        bifurcation_prob = HDBifurcationPDF(mask.numpy())
        results = []
        for x,y in coords:
            bif_prob = bifurcation_prob(x,y)
            croped = image[int(x - 0.5*LesionDescriber.BOX_SIZE):int(x + 0.5*LesionDescriber.BOX_SIZE), 
                           int(y - 0.5*LesionDescriber.BOX_SIZE):int(y + 0.5*LesionDescriber.BOX_SIZE)].unsqueeze(0)
            prob = self.model(croped)

            results.append({
                'BIFURCATION':bif_prob,
                'AORTO_OSTIAL_STENOSIS':prob[0],
                'BLUNT_STUMP':prob[1],
                'BRIDGING':prob[2],
                'HEAVY_CALCIFICATION':prob[3],
                'SEVERE_TORTUOSITY':prob[4], 
                'THROMBUS':prob[5],
            })
        return results
    
    def predict(self,image:torch.Tensor,mask:torch.Tensor,coords:Union[int,int]) -> dict[str:float]:
        bifurcation_prob = HDBifurcationPDF(mask.numpy())
        x,y = coords
        bif_prob = bifurcation_prob(x,y)
        croped = image[int(x - 0.5*LesionDescriber.BOX_SIZE):int(x + 0.5*LesionDescriber.BOX_SIZE), 
                       int(y - 0.5*LesionDescriber.BOX_SIZE):int(y + 0.5*LesionDescriber.BOX_SIZE)].unsqueeze(0)
        prob = self.model(croped)

        return {
                'BIFURCATION':bif_prob,
                'AORTO_OSTIAL_STENOSIS':prob[0],
                'BLUNT_STUMP':prob[1],
                'BRIDGING':prob[2],
                'HEAVY_CALCIFICATION':prob[3],
                'SEVERE_TORTUOSITY':prob[4], 
                'THROMBUS':prob[5],
            }

        