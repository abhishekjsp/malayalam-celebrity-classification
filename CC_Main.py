import Data_Preprocessor as dp
import Model_Trainer as mt

CLEAN_DATA = False

if CLEAN_DATA:
    dp.clean_data('images', 'cropped_imgs')

X_data, Y_data = dp.prepare_dataset('cleaned_imgs')

mt.train_model_and_predict(X_data, Y_data)
