# MATQA
Multi-Answer Time-Series Knowledge Reasoning


Subsequent source code will be uploaded one after another after combing.





运行：
  测试：
	Detect/KGDOld/new_test.py
  训练：
    Detect/proposed/ModelProposed_1.py

     1、取消注释
         # if __name__ == "__main__":
         #     test_model = ProposedModel()
         #     test_model.train()
	  2、Config.batchsize由1设置成32





We thank zhen jia et al. for their advice and inspiration on this work in the early stage, while the details of the dataset can be found in the following literature.

@article{jia2021complex,
  title={Complex Temporal Question Answering on Knowledge Graphs},
  author={Jia, Zhen and Pramanik, Soumajit and Roy, Rishiraj Saha and Weikum, Gerhard},
  journal={arXiv preprint arXiv:2109.08935},
  year={2021}
}

or 

https://exaqt.mpi-inf.mpg.de/



We are equally grateful to Michihiro et al. for inspiring us with their proposed GNN model. Also, citations can be made using the following format

Yasunaga, M., Ren, H., Bosselut, A., Liang, P., Leskovec, J., 2021. QA-GNN: Reasoning with Language Models and Knowledge Graphs for Question Answering. https://doi.org/10.48550/arXiv.2104.06378


