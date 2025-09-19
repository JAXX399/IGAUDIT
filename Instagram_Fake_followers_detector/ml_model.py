import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier

# Path to the model file
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, 'rfc_model.pkl')

def train_model():
    """
    Train the Random Forest Classifier model using the training data.
    This would normally be run once and the model saved.
    """
    # Load training data
    train = pd.read_csv(os.path.join(MODEL_DIR, "train.csv"))
    
    # Split into features and labels
    train_Y = train.fake
    train_X = train.drop(columns='fake')
    
    # Train the model
    rfc = RandomForestClassifier()
    model = rfc.fit(train_X, train_Y)
    
    # Save the model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    return model

def load_model():
    """
    Load the trained model from disk, or train a new one if it doesn't exist.
    """
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    except (FileNotFoundError, EOFError):
        print("Model not found, training a new one...")
        return train_model()

def prepare_follower_features(follower_info):
    """
    Extract and format features from a follower's info to match the model's expected input.
    Based on the feature extraction in the notebook.
    """
    # Extract basic features
    profile_pic = 1 if follower_info.get('profile_pic_url') else 0
    username = follower_info.get('username', '')
    fullname = follower_info.get('full_name', '')
    
    # Calculate derived features
    username_length = len(username)
    nums_in_username = sum(c.isdigit() for c in username)
    nums_username_ratio = nums_in_username / username_length if username_length > 0 else 0
    
    fullname_words = len(fullname.split()) if fullname else 0
    fullname_length = len(fullname)
    nums_in_fullname = sum(c.isdigit() for c in fullname)
    nums_fullname_ratio = nums_in_fullname / fullname_length if fullname_length > 0 else 0
    
    name_equals_username = 1 if username.lower() == fullname.replace(' ', '').lower() else 0
    
    description_length = len(follower_info.get('biography', ''))
    external_url = 1 if follower_info.get('external_url') else 0
    is_private = 1 if follower_info.get('is_private') else 0
    
    media_count = follower_info.get('media_count', 0)
    follower_count = follower_info.get('follower_count', 0)
    following_count = follower_info.get('following_count', 0)
    
    # Return as a dictionary matching the training data columns
    return {
        'profile pic': profile_pic,
        'nums/length username': nums_username_ratio,
        'fullname words': fullname_words,
        'nums/length fullname': nums_fullname_ratio,
        'name==username': name_equals_username,
        'description length': description_length,
        'external URL': external_url,
        'private': is_private,
        '#posts': media_count,
        '#followers': follower_count,
        '#follows': following_count
    }

def predict_fake_followers(followers_info):
    """
    Predict which followers are fake using the trained model.
    
    Args:
        followers_info: List of follower information dictionaries
        
    Returns:
        List of 0/1 predictions (0=authentic, 1=fake)
    """
    # Load the model
    model = load_model()
    
    # Prepare features for each follower
    features_list = [prepare_follower_features(info) for info in followers_info]
    
    # Convert to DataFrame in the right format
    features_df = pd.DataFrame(features_list)
    
    # Make predictions
    predictions = model.predict(features_df)
    
    return predictions.tolist()
