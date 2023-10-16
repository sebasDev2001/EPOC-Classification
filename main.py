import scripts.merge_model as merge

GB_DIR = "./models/StandarModels/GradientBoost_model.pkl"
NN_DIR = "./models/CNN/resnet_18_finetuning.pth"

def main() : 
    model = merge.MergeModel(GB_DIR, NN_DIR, nn_weight=0.50, gb_weight=0.50)
    model.eval_model()

if __name__ == '__main__':
    main()
