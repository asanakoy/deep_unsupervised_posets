# Deep Unsupervised Similarity Learning Using Partially Ordered Sets (CVPR 2017)

Accepted at CVPR 2017   
**"Deep Unsupervised Similarity Learning Using Partially Ordered Sets"** 
Miguel A. Bautista* , Artsiom Sanakoyeu* , Björn Ommer.

---

* Paper: https://arxiv.org/abs/1704.02268
* GT labels for Olympic Sports dataset: [olympic_sports_retrieval/data](https://github.com/asanakoy/cliquecnn/tree/master/olympic_sports_retrieval/data)
* Evaluation script for Olympic Sports dataset:
[calculate_roc_auc.py](https://github.com/asanakoy/cliquecnn/blob/master/olympic_sports_retrieval/calculate_roc_auc.py)
* Baseline HOG-LDA similarity matrices for Olympic Sports:
[similarities_hog_lda.tar.zip](http://compvis10.iwr.uni-heidelberg.de/share/cliquecnn/similarities_hog_lda.tar.zip) (11.5 Gb)


### Tensorflow models for Olympic Sports dataset trained with our approach

All models were trained from scratch **without** *Imagenet pretraining* and **without** *any supervision*.   

1. The model trained on all frames from Olympic sports dataset: [olympic_sports_all_cat_convnet_scratch_strip.ckpt](https://hcicloud.iwr.uni-heidelberg.de/index.php/s/Eryj3TB9quLGYtz)   
2. Using the same method we finetuned the previous model for each sport independently w/o any supervision (we again used only grouping and posets that we build without GT information).   
Single models for each sport: [olympic_sports_models_from_scratch](https://hcicloud.iwr.uni-heidelberg.de/index.php/s/z35nuNm76p1v9pR)



### Requirements
- Python 2.7
- Tensorflow r1.*

### Example
Example how to load models: [example_load_networks.ipynb](example_load_networks.ipynb).


---

If you find this code or data useful for your research, please cite
```
@inproceedings{UnsupSimPosets2017,
  title={Deep Unsupervised Similarity Learning using Partially Ordered Sets}
  author={Bautista, Miguel A and Sanakoyeu, Artsiom and Ommer, Bj{\"o}rn},
  booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}
```
