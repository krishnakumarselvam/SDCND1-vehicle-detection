from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import json
from lesson_functions import *



class svm_predictor():
    def __init__(self, model_config_path, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(model_config_path) as f:
            self.config = json.load(f)

    def preprocess(self, image, precomputed_hog_features=[]):
        color_space = self.config['color_space']
        spatial_size=(self.config['spatial_size'][0], self.config['spatial_size'][1])
        hist_bins=self.config['hist_bins']
        orient=self.config['orient']
        pix_per_cell=self.config['pix_per_cell']
        cell_per_block=self.config['cell_per_block']
        hog_channel=self.config.get('hog_channel')
        spatial_feat=self.config['spatial_feat']
        hist_feat=self.config['hist_feat']
        hog_feat=self.config['hog_feat']

        file_features = []
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            if len(precomputed_hog_features):
                hog_features = precomputed_hog_features
            else:
                if hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                            orient, pix_per_cell, cell_per_block, 
                                            vis=False, feature_vec=True))
                    hog_features = np.ravel(hog_features)        
                else:
                    hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)                
            # Append the new feature vector to the features list
            file_features.append(hog_features)

        return np.concatenate(file_features)

    def predict(self, img, precomputed_hog_features=[]):
        raw_features = self.preprocess(img, precomputed_hog_features)
        return self._predict(raw_features)

    def _predict(self, raw_features, single_value = True):
        if single_value:
            scaled_features = self.scaler.transform([raw_features]).reshape(1, -1)
            return self.model.predict(scaled_features)
        else:
            return None # Todo - implement the vectorized version

def get_predictor(version):
    basepath = 'models/'
    return svm_predictor(
        '{}model_config_{}.json'.format(basepath, version),
        '{}model_{}.pkl'.format(basepath, version),
        '{}X_scaler_{}.pkl'.format(basepath, version))

class ensemble_predictor():
    def __init__(self, versions=[1]):
        self.versions = versions
        self.predictors = []
        for v in versions:
            model_config_path, model_path, scaler_path = self._get_paths(v)
            self.predictors.append(svm_predictor(model_config_path, model_path, scaler_path))

    def _get_paths(self, version):
        basepath = 'models/'
        return 'models/model_config_{}.pkl'.format(version), 'models/model_{}.pkl'.format(version), 'models/X_scaler_{}.pkl'.format(version)

    def predict(self, img):
        results = [m.predict]
