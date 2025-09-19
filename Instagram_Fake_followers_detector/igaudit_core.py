from instagram_private_api import Client, ClientCompatPatch
import random
import sys
import os

# Import the ML model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml_model import predict_fake_followers, prepare_follower_features

def get_ID(api, username):
    return api.username_info(username)['user']['pk']

def get_followers(api, user_id, rank):
    followers = []
    next_max_id = True
    while next_max_id:
        if next_max_id is True:
            next_max_id = ''
        f = api.user_followers(user_id, rank, max_id=next_max_id)
        followers.extend(f.get('users', []))
        next_max_id = f.get('next_max_id', '')
    user_fer = [dic['username'] for dic in followers]
    return user_fer

def get_follower_info(api, username):
    user_id = get_ID(api, username)
    return api.user_info(user_id)['user']

def get_user_posts(api, user_id, rank):
    """Get user posts for engagement analysis"""
    posts = []
    next_max_id = True
    
    while next_max_id and len(posts) < 20:  # Limit to 20 posts for efficiency
        if next_max_id is True:
            next_max_id = ''
        feed = api.user_feed(user_id, rank, max_id=next_max_id)
        posts.extend(feed.get('items', []))
        next_max_id = feed.get('next_max_id', '')
    
    return posts

def calculate_engagement_rate(posts, follower_count):
    """Calculate engagement rate based on likes and comments"""
    if not posts or follower_count == 0:
        return 0.0
    
    total_likes = sum(post.get('like_count', 0) for post in posts)
    total_comments = sum(post.get('comment_count', 0) for post in posts)
    total_engagement = total_likes + total_comments
    
    avg_engagement_per_post = total_engagement / len(posts)
    engagement_rate = (avg_engagement_per_post / follower_count) * 100
    
    return engagement_rate

def run_audit(username: str, password: str, target_username: str = None):
    result = {}
    try:
        # Login to Instagram
        api = Client(username, password)
        if target_username is None:
            target_username = username
        
        # Get user info
        user_id = get_ID(api, target_username)
        user_info = api.user_info(user_id)['user']
        rank = api.generate_uuid()
        
        # Get followers
        followers = get_followers(api, user_id, rank)
        sample_size = min(50, len(followers))
        if sample_size == 0:
            raise Exception("No followers found.")
        
        # Sample followers and get their info
        random_followers = random.sample(followers, sample_size)
        f_infos = [get_follower_info(api, f) for f in random_followers]
        
        # Use ML model to predict fake followers
        fake_labels = predict_fake_followers(f_infos)
        no_fakes = sum(fake_labels)
        authenticity = ((sample_size - no_fakes) * 100) / sample_size
        
        # Get posts and calculate engagement rate
        posts = get_user_posts(api, user_id, rank)
        follower_count = user_info.get('follower_count', 0)
        engagement_rate = calculate_engagement_rate(posts, follower_count)
        
        # Prepare result
        result['username'] = target_username
        result['user_info'] = {
            'follower_count': user_info.get('follower_count'),
            'following_count': user_info.get('following_count'),
            'media_count': user_info.get('media_count'),
            'is_private': user_info.get('is_private'),
            'full_name': user_info.get('full_name'),
            'bio': user_info.get('biography'),
        }
        result['audit'] = {
            'sampled_followers': sample_size,
            'no_fakes': no_fakes,
            'authenticity_percent': authenticity,
            'engagement_rate': engagement_rate,
            'posts_analyzed': len(posts)
        }
        result['status'] = 'success'
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
    return result
