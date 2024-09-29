class VideoPipeline:
    def __init__(self, csv_box_name, save_path, img_folder_name,base_results_folder, metadata=None):
        self.csv_box_name = csv_box_name
        self.save_path = save_path
        self.img_folder_name = img_folder_name
        self.base_results_folder = base_results_folder
        self.metadata = metadata