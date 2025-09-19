import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from playwright.sync_api import sync_playwright
import time
import json
import random
import os
import re

# Path to training data
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MODEL_DIR, 'data')
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.csv')

# Initialize the model
rfc_model = None


def get_model():
    """
    Get the Random Forest Classifier model, training it if needed.
    """
    global rfc_model
    if rfc_model is None:
        # Load training data
        try:
            train = pd.read_csv(TRAIN_DATA_PATH)
        except FileNotFoundError:
            # If training data doesn't exist, create a simple model with default data
            print("Training data not found, using default model")
            # Create a simple dataset for fake account detection
            data = {
                'profile pic': [1, 1, 1, 1, 0, 0],
                'nums/length username': [0.1, 0.0, 0.0, 0.0, 0.5, 0.8],
                'fullname words': [2, 3, 2, 2, 0, 0],
                'nums/length fullname': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'name==username': [0, 0, 0, 0, 1, 1],
                'description length': [100, 150, 50, 200, 0, 0],
                'external URL': [1, 1, 0, 1, 0, 0],
                'private': [0, 0, 0, 0, 1, 1],
                '#posts': [50, 100, 30, 200, 0, 0],
                '#followers': [1000, 2000, 500, 5000, 10, 5],
                '#follows': [500, 800, 300, 1000, 1000, 2000],
                'fake': [0, 0, 0, 0, 1, 1]
            }
            train = pd.DataFrame(data)

        # Split into features and labels
        train_Y = train.fake
        train_X = train.drop(columns='fake')

        # Train the model - Random Forest had the best accuracy in the notebook
        rfc = RandomForestClassifier()
        rfc_model = rfc.fit(train_X, train_Y)

    return rfc_model


def prepare_follower_features(user_info):
    """
    Extract features from a user's info to match the model's expected input.
    Based on the feature extraction in the notebook.
    """
    # Extract basic features
    has_profile_pic = 1 if user_info.get('profile_pic_url') else 0

    # Username features
    username = user_info.get('username', '')
    username_length = len(username)
    nums_in_username = sum(c.isdigit() for c in username)
    nums_username_ratio = nums_in_username / \
        username_length if username_length > 0 else 0

    # Fullname features
    fullname = user_info.get('full_name', '')
    fullname_words = len(fullname.split()) if fullname else 0
    fullname_length = len(fullname)
    nums_in_fullname = sum(c.isdigit() for c in fullname)
    nums_fullname_ratio = nums_in_fullname / \
        fullname_length if fullname_length > 0 else 0

    # Other features
    name_equals_username = 1 if username.lower(
    ) == fullname.replace(' ', '').lower() else 0
    description_length = len(user_info.get('biography', ''))
    external_url = 1 if user_info.get('external_url') else 0
    is_private = 1 if user_info.get('is_private') else 0

    # Count features
    media_count = user_info.get('media_count', 0)
    follower_count = user_info.get('follower_count', 0)
    following_count = user_info.get('following_count', 0)

    # Return as a dictionary matching the training data columns
    return {
        'profile pic': has_profile_pic,
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
    # Get the model
    model = get_model()

    # Prepare features for each follower
    features_list = [prepare_follower_features(
        info) for info in followers_info]

    # Convert to DataFrame in the right format
    features_df = pd.DataFrame(features_list)

    # Make predictions
    predictions = model.predict(features_df)

    return predictions.tolist()


def get_instagram_client(username, password):
    """
    Create an Instagram client using Playwright browser automation.
    Opens the browser for manual login but automates the scraping.

    Args:
        username: Instagram username (optional, for reference only)
        password: Instagram password (optional, for reference only)

    Returns:
        Playwright browser context with logged-in Instagram session
    """
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context(
        viewport={"width": 1280, "height": 800},
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
    page = context.new_page()

    # Navigate to Instagram login page
    page.goto("https://www.instagram.com/accounts/login/")

    # Prompt user to manually log in
    print("\n\n========== MANUAL LOGIN REQUIRED ==========")
    print("A browser window has been opened to Instagram.")
    print("Please log in manually with your credentials.")
    print("The program will continue automatically after you log in.")
    print("===========================================\n")

    # Wait for user to manually log in (detect successful login)
    try:
        # Wait for the Instagram feed to load (indicating successful login)
        # This has a long timeout to give the user plenty of time to log in manually
        page.wait_for_selector(
            # 5 minutes timeout
            'svg[aria-label="Home"]', state="visible", timeout=300000)
        print("\nLogin successful! Continuing with automated scraping...")
    except Exception as e:
        print(f"Login timeout or error: {e}")
        browser.close()
        playwright.stop()
        raise Exception("Manual login failed or timed out. Please try again.")

    return {"page": page, "context": context, "browser": browser, "playwright": playwright}


def close_instagram_client(client):
    """Close the Playwright browser and resources"""
    client["browser"].close()
    client["playwright"].stop()


def get_user_data_from_page(page, username):
    """Extract user data from their profile page using direct JavaScript evaluation"""
    # Retry mechanism with exponential backoff
    max_retries = 5
    base_timeout = 60000  # 60 seconds

    for attempt in range(max_retries):
        current_timeout = base_timeout * \
            (1.5 ** attempt)  # Exponential backoff
        try:
            print(
                f"  Attempt {attempt+1}/{max_retries} to load profile (timeout: {int(current_timeout/1000)}s)...")

            # Navigate to the user's profile with increased timeout
            page.goto(
                f"https://www.instagram.com/{username}/", timeout=current_timeout)

            # Try different wait strategies
            try:
                # First try waiting for DOM content (faster than networkidle)
                page.wait_for_load_state("domcontentloaded", timeout=30000)
            except Exception as e:
                print(f"  DOM content load timeout: {e}")

            # Wait a bit for JavaScript to execute
            page.wait_for_timeout(5000)

            # Use JavaScript to extract data directly from the page
            # This is more reliable than using selectors which can change
            user_data = page.evaluate('''() => {
                // Check if page doesn't exist
                if (document.body.innerText.includes("Sorry, this page isn't available")) {
                    return { exists: false };
                }

                // Check if account is private
                const isPrivate = document.body.innerText.includes("This Account is Private");

                // Get meta data
                let followerCount = 0;
                let followingCount = 0;
                let postsCount = 0;

                // Try to get counts from meta section
                const metaItems = document.querySelectorAll('header section ul li');
                if (metaItems.length >= 3) {
                    // Extract numbers using regex
                    const postsText = metaItems[0].innerText;
                    const postsMatch = postsText.match(/(\\d+(?:,\\d+)*)/);
                    postsCount = postsMatch ? parseInt(postsMatch[0].replace(/,/g, '')) : 0;

                    const followersText = metaItems[1].innerText;
                    const followersMatch = followersText.match(/(\\d+(?:,\\d+)*)/);
                    followerCount = followersMatch ? parseInt(followersMatch[0].replace(/,/g, '')) : 0;

                    const followingText = metaItems[2].innerText;
                    const followingMatch = followingText.match(/(\\d+(?:,\\d+)*)/);
                    followingCount = followingMatch ? parseInt(followingMatch[0].replace(/,/g, '')) : 0;
                }

                // Get profile pic
                let profilePicUrl = null;
                const imgElement = document.querySelector('header img');
                if (imgElement) {
                    profilePicUrl = imgElement.src;
                }

                // Get full name
                let fullName = '';
                const nameElement = document.querySelector('header section h2');
                if (nameElement) {
                    fullName = nameElement.innerText;
                }

                // Get bio
                let bio = '';
                const bioElement = document.querySelector('header section > div > span');
                if (bioElement) {
                    bio = bioElement.innerText;
                }

                // Get external URL
                let externalUrl = null;
                const urlElement = document.querySelector('header section a[target="_blank"]');
                if (urlElement) {
                    externalUrl = urlElement.href;
                }

                return {
                    exists: true,
                    is_private: isPrivate,
                    follower_count: followerCount,
                    following_count: followingCount,
                    media_count: postsCount,
                    profile_pic_url: profilePicUrl,
                    full_name: fullName,
                    biography: bio,
                    external_url: externalUrl
                };
            }''')

            # Check if page exists
            if not user_data.get('exists', True):
                print(f"  Profile for {username} doesn't exist")
                return {
                    'username': username,
                    'full_name': username,
                    'profile_pic_url': None,
                    'biography': '',
                    'external_url': None,
                    'is_private': False,
                    'media_count': 0,
                    'follower_count': 0,
                    'following_count': 0,
                    'pk': username,
                    'exists': False
                }

            # Construct user info object
            user_info = {
                'username': username,
                'full_name': user_data.get('full_name', username),
                'profile_pic_url': user_data.get('profile_pic_url'),
                'biography': user_data.get('biography', ''),
                'external_url': user_data.get('external_url'),
                'is_private': user_data.get('is_private', False),
                'media_count': user_data.get('media_count', 0),
                'follower_count': user_data.get('follower_count', 0),
                'following_count': user_data.get('following_count', 0),
                'pk': username,  # Using username as a stand-in for user ID
                'exists': True
            }

            print(f"  Successfully extracted profile data for {username}")
            print(
                f"  Followers: {user_info['follower_count']}, Following: {user_info['following_count']}, Posts: {user_info['media_count']}")
            return user_info

        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)  # Progressive backoff
                print(f"  Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)

    # If all retries fail, return a minimal user info object with default values
    print(
        f"  All attempts to get user data for {username} failed, using default values")
    return {
        'username': username,
        'full_name': username,
        'profile_pic_url': None,
        'biography': '',
        'external_url': None,
        'is_private': False,
        'media_count': 0,
        'follower_count': 0,
        'following_count': 0,
        'pk': username,
        'exists': False
    }


def get_followers_data(client, username, max_followers=50):
    """
    Get followers data by clicking on the followers button and scraping from the dialog

    Args:
        client: Playwright browser context
        username: Target username
        max_followers: Maximum number of followers to scrape

    Returns:
        List of follower usernames
    """
    page = client["page"]

    try:
        # Click the followers button
        print(f"\nClicking followers button for {username}...")
        followers_button = page.wait_for_selector(
            'a[href$=\"/followers/\"]', timeout=10000)
        followers_button.click()

        # Wait for the dialog to appear
        print("\nWaiting for followers dialog...")
        page.wait_for_selector('div[role=\"dialog\"]', timeout=10000)

        print("\nPlease scroll through the followers list manually until you see all the followers you want to analyze.")
        print(
            f"You can scroll until you see at least {max_followers} followers.")
        print("When you're done scrolling, press Enter to continue...")

        # Wait for user input
        input("Press Enter when you're done scrolling...")

        # Get all visible follower elements
        print("\nExtracting visible followers...")
        try:
            # Use a more reliable selector with proper escaping
            follower_elements = page.query_selector_all(
                'div[role=\"dialog\"] a:not([href*=\"/p/\"]):not([href*=\"/explore/\"]):not([href*=\"/reels/\"]):not([href*=\"/stories/\"]):not([href*=\"/direct/\"]):not([href*=\"/tv/\"]):not([href*=\"/guides/\"]):not([href*=\"/highlights/\"])')

            print(
                f"Found {len(follower_elements)} potential follower elements")
            followers = []
            for element in follower_elements:
                try:
                    href = element.get_attribute('href')
                    if href:
                        # Extract username from href
                        username = href.replace('/', '').split('/')[0]
                        if username and username not in followers:
                            print(f"Found new follower: {username}")
                            followers.append(username)
                except Exception as e:
                    print(f"Error processing element: {e}")
        except Exception as e:
            print(f"Error getting follower elements: {e}")
            # Try a simpler selector as fallback
            try:
                print("\nTrying simpler selector as fallback...")
                follower_elements = page.query_selector_all(
                    'div[role=\"dialog\"] a')
                print(
                    f"Found {len(follower_elements)} elements with simpler selector")

                for element in follower_elements:
                    try:
                        href = element.get_attribute('href')
                        if href and not any(x in href for x in ['/p/', '/explore/', '/reels/', '/stories/', '/direct/', '/tv/', '/guides/', '/highlights/']):
                            # Extract username from href
                            username = href.replace('/', '').split('/')[0]
                            if username and username not in followers:
                                print(f"Found new follower: {username}")
                                followers.append(username)
                    except Exception as e:
                        print(f"Error processing element: {e}")
            except Exception as e:
                print(f"Error with simpler selector: {e}")

        # Filter out system accounts and the target user
        print("\nFiltering out system accounts and target user...")
        system_accounts = ['web', 'legal', 'direct', 'tv',
                           'reels', 'stories', 'guides', 'highlights']
        filtered_followers = [
            f for f in followers
            if f.lower() != username.lower() and
            f.lower() not in system_accounts and
            any(c.isalnum() for c in f)
        ]

        if filtered_followers:
            print(
                f"\nFound {len(filtered_followers)} valid followers")
            return filtered_followers[:max_followers]
        else:
            print(
                "\nNo valid followers found after filtering system accounts")
            return []

    except Exception as e:
        print(f"\nError getting followers: {e}")
        return []


def run_audit(username, password, target_username=None):
    """
    Run a complete Instagram audit using the model from the notebook.

    Args:
        username: Instagram username for reference (manual login will be required)
        password: Instagram password for reference (manual login will be required)
        target_username: Username to audit (if different from login)

    Returns:
        Dictionary with audit results
    """
    result = {}
    client = None

    try:
        # Login to Instagram using Playwright (manual login)
        client = get_instagram_client(username, password)

        if target_username is None:
            # Since we don't know who logged in, ask for the target username
            if not target_username:
                print("\nWho would you like to audit?")
                target_username = input(
                    "Enter Instagram username to audit: ")

        print(f"\n===== Starting audit for {target_username} =====")

        # Get user info - this is the most important part
        print(f"\nGetting profile data for {target_username}...")
        user_info = get_user_data_from_page(client["page"], target_username)

        # Check if user exists
        if not user_info.get('exists', True):
            return {
                'status': 'error',
                'error': f"User {target_username} doesn't exist"
            }

        # Try to get followers, but don't fail the whole audit if we can't
        print(f"\nGetting followers data for {target_username}...")
        followers = get_followers_data(client, target_username)

        # If we can't get followers or there aren't any, do a limited audit
        if not followers or len(followers) == 0:
            print(
                f"No followers found or account is private. Limited audit will be performed.")

            # Calculate engagement metrics if possible
            engagement_rate = 0
            posts_analyzed = 0

            # Create a basic result with just the user info
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
                'follower_analysis': {
                    'sampled_followers': 0,
                    'fake_followers': 0,
                    'authenticity_percent': 0,
                },
                'engagement_analysis': {
                    'engagement_rate': engagement_rate,
                    'posts_analyzed': posts_analyzed,
                }
            }
            result['status'] = 'partial'
            result['message'] = "Limited audit performed: only basic profile information available"
            return result

        # If we have followers, try to analyze them
        sample_size = min(50, len(followers))
        print(f"\nAnalyzing {sample_size} followers...")

        # Sample followers and get their info
        random_followers = random.sample(followers, sample_size)
        f_infos = []

        for i, f in enumerate(random_followers):
            print(f"Processing follower {i+1}/{sample_size}: {f}")
            f_info = get_user_data_from_page(client["page"], f)
            f_infos.append(f_info)

            # Add a small delay between requests to avoid rate limiting
            if i < sample_size - 1:  # Don't delay after the last one
                time.sleep(1)

        # Use ML model to predict fake followers
        print("\nPredicting fake followers...")
        fake_labels = predict_fake_followers(f_infos)
        no_fakes = sum(fake_labels)
        authenticity = ((sample_size - no_fakes) * 100) / \
            sample_size if sample_size > 0 else 0

        # Track fake follower usernames
        fake_follower_usernames = []
        for i, is_fake in enumerate(fake_labels):
            if is_fake == 1:
                fake_follower_usernames.append(random_followers[i])

        # Prepare result
        print("\nPreparing audit results...")
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
            'follower_analysis': {
                'sampled_followers': sample_size,
                'fake_followers': no_fakes,
                'authenticity_percent': authenticity,
                'fake_follower_usernames': fake_follower_usernames,
            },
            'engagement_analysis': {
                'engagement_rate': 0,  # Simplified version doesn't calculate engagement
                'posts_analyzed': 0,
            }
        }

        result['status'] = 'success'
        print("\n===== Audit completed successfully =====")

    except Exception as e:
        print(f"\nError during audit: {e}")
        result['status'] = 'error'
        result['error'] = str(e)
    finally:
        # Close the browser
        if client:
            print("\nClosing browser...")
            close_instagram_client(client)

    return result
