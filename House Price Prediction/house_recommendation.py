# house_recommendation.py
import pandas as pd
import numpy as np

def load_data():
    """Load your house data"""
    df = pd.read_csv('data.csv')
    return df

def get_user_preferences():
    """Get house preferences from user"""
    print("🏠HOUSE RECOMMENDATION SYSTEM")
    print("=" * 40)
    
    preferences = {}
    
    # Budget
    preferences['min_price'] = float(input("Minimum budget ($): ") or 0)
    preferences['max_price'] = float(input("Maximum budget ($): ") or 500000)
    
    # Area preferences
    df = load_data()
    print("\nAvailable neighborhoods:", df['Neighborhood'].unique())
    area_input = input("Preferred neighborhoods (comma-separated, or press enter for all): ")
    preferences['preferred_areas'] = [area.strip() for area in area_input.split(',')] if area_input else []
    
    # House features
    preferences['min_bedrooms'] = int(input("Minimum bedrooms: ") or 0)
    preferences['min_bathrooms'] = int(input("Minimum full bathrooms: ") or 0)
    preferences['min_living_area'] = float(input("Minimum living area (sq ft): ") or 0)
    
    # Optional features
    preferences['must_have_garage'] = input("Must have garage? (y/n): ").lower() == 'y'
    preferences['min_quality'] = int(input("Minimum overall quality (1-10, 0 for any): ") or 0)
    
    return preferences

def filter_houses(df, preferences):
    """Filter houses that match user preferences"""
    filtered = df.copy()
    
    print(f"\n FILTERING HOUSES...")
    print(f"Total houses in dataset: {len(df)}")
    
    # Price filter
    filtered = filtered[
        (filtered['SalePrice'] >= preferences['min_price']) & 
        (filtered['SalePrice'] <= preferences['max_price'])
    ]
    print(f"After price filter: {len(filtered)}")
    
    # Neighborhood filter
    if preferences['preferred_areas']:
        filtered = filtered[filtered['Neighborhood'].isin(preferences['preferred_areas'])]
        print(f"After area filter: {len(filtered)}")
    
    # Bedrooms filter
    if preferences['min_bedrooms'] > 0:
        filtered = filtered[filtered['BedroomAbvGr'] >= preferences['min_bedrooms']]
        print(f"After bedrooms filter: {len(filtered)}")
    
    # Bathrooms filter
    if preferences['min_bathrooms'] > 0:
        filtered = filtered[filtered['FullBath'] >= preferences['min_bathrooms']]
        print(f"After bathrooms filter: {len(filtered)}")
    
    # Living area filter
    if preferences['min_living_area'] > 0:
        filtered = filtered[filtered['GrLivArea'] >= preferences['min_living_area']]
        print(f"After living area filter: {len(filtered)}")
    
    # Garage filter
    if preferences['must_have_garage']:
        filtered = filtered[filtered['GarageCars'] > 0]
        print(f"After garage filter: {len(filtered)}")
    
    # Quality filter
    if preferences['min_quality'] > 0:
        filtered = filtered[filtered['OverallQual'] >= preferences['min_quality']]
        print(f"After quality filter: {len(filtered)}")
    
    return filtered

def score_and_rank_houses(houses_df, preferences):
    """Score houses based on how well they match preferences"""
    if len(houses_df) == 0:
        return houses_df
    
    scored_houses = houses_df.copy()
    scores = []
    
    for idx, house in scored_houses.iterrows():
        score = 0
        
        # Price score (lower price = higher score within budget)
        price_range = preferences['max_price'] - preferences['min_price']
        if price_range > 0:
            price_score = 100 * (1 - (house['SalePrice'] - preferences['min_price']) / price_range)
            score += price_score * 0.3
        
        # Size score
        if preferences['min_living_area'] > 0:
            size_bonus = min(50, (house['GrLivArea'] - preferences['min_living_area']) / 100)
            score += size_bonus
        
        # Quality score
        quality_bonus = house['OverallQual'] * 3
        score += quality_bonus
        
        # Bedroom bonus
        if preferences['min_bedrooms'] > 0:
            bedroom_bonus = (house['BedroomAbvGr'] - preferences['min_bedrooms']) * 5
            score += max(0, bedroom_bonus)
        
        # Bathroom bonus
        if preferences['min_bathrooms'] > 0:
            bathroom_bonus = (house['FullBath'] - preferences['min_bathrooms']) * 8
            score += max(0, bathroom_bonus)
        
        # Garage bonus
        if preferences['must_have_garage'] and house['GarageCars'] > 0:
            score += house['GarageCars'] * 5
        
        scores.append(score)
    
    scored_houses['RecommendationScore'] = scores
    scored_houses = scored_houses.sort_values('RecommendationScore', ascending=False)
    
    return scored_houses

def display_recommendations(ranked_houses, top_n=5):
    """Display top recommended houses"""
    if len(ranked_houses) == 0:
        print("No houses found matching your criteria. Try relaxing some requirements.")
        return
    
    print(f"\n🎯 TOP {min(top_n, len(ranked_houses))} RECOMMENDED HOUSES")
    print("=" * 60)
    
    for i, (idx, house) in enumerate(ranked_houses.head(top_n).iterrows(), 1):
        print(f"\n#{i} | Score: {house['RecommendationScore']:.1f}/100")
        print(f" Price: ${house['SalePrice']:,.2f}")
        print(f" Neighborhood: {house['Neighborhood']}")
        print(f" Living Area: {house['GrLivArea']} sq ft")
        print(f" Bedrooms: {house['BedroomAbvGr']} | 🛁 Full Baths: {house['FullBath']}")
        print(f" Quality: {house['OverallQual']}/10")
        print(f" Garage: {house['GarageCars']} cars")
        
        # Show why it's recommended
        print("   Why recommended:")
        if house['SalePrice'] < ranked_houses['SalePrice'].median():
            print("   - Good value for money")
        if house['GrLivArea'] > ranked_houses['GrLivArea'].median():
            print("   - Larger than average")
        if house['OverallQual'] >= 7:
            print("   - High quality construction")

def run_recommendation_system():
    """Run the complete house recommendation system"""
    print("🏠 SMART HOUSE RECOMMENDATION ENGINE")
    print("=" * 50)
    
    # Load data
    df = load_data()
    
    # Step 1: Get user preferences
    user_prefs = get_user_preferences()
    
    # Step 2: Filter houses
    matching_houses = filter_houses(df, user_prefs)
    
    if len(matching_houses) == 0:
        print("No houses match your criteria. Try adjusting your preferences.")
        return
    
    # Step 3: Score and rank
    ranked_houses = score_and_rank_houses(matching_houses, user_prefs)
    
    # Step 4: Display results
    display_recommendations(ranked_houses)
    
    print(f"\Recommendation complete! Found {len(matching_houses)} matching houses.")

# Run the system when this file is executed directly
if __name__ == "__main__":
    run_recommendation_system()