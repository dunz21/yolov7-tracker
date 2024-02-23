from utils.pipeline import getFinalScore


FOLDER_NAME = "/home/diego/Documents/yolov7-tracker/imgs_conce"
features_file = "conce_solider_in-out_DB.csv"
distance_file = "conce_distance_cosine_DB.csv"
html_file = "conce_cosine_match_DB_Final.html"

solider = {
    'name': "solider",
    'weights': "/home/diego/Documents/detectron2/solider_model.pth"
}

alignedReID = {
    'name': "alignedReID",
    'weights': "/home/diego/Documents/AlignedReID/log_marketTOTAL_MODIFIED_global_local_labelSmooth/best_model.pth.tar"
}


MODEL = alignedReID['name']
WEIGHTS = alignedReID['weights']

getFinalScore(folder_name=FOLDER_NAME,model=MODEL,features_file=features_file,distance_file=distance_file,html_file=html_file,distance_method="cosine",weights=WEIGHTS)