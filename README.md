# SCNet-Lightning

A re-implementation of the (unofficial) SCNet audio source separation model using Pytorch Lightning to facilitate debugging and reproduction.

You can find the pure pytorch implementation in https://github.com/amanteur/SCNet-PyTorch 
and the original model description in [this paper](https://arxiv.org/abs/2401.13276).

### Model overview
<img width="878" height="574" alt="architecture" src="https://github.com/user-attachments/assets/82046aa2-57ac-467c-9e21-9c9bef52051d" />


### Experimental Results
While I was unable to reach the same metrics as the original paper, they're still pretty good

|             | **SDR** |
|-------------|---------|
| **Vocals**  |    9.44 |
| **Bass**    |    8.14 |
| **Drums**   |   10.05 |
| **Other**   |    6.40 |
| **Overall** |    8.48 |


### To-do's
- [ ] Train with additional data.
- [ ] Write a proper tutorial.
- [ ] Create an end-user script for audio separation.