"""
Session 1: TensorFlow Fundamentals & Basic Neural Networks
3 Projects + 1 Classwork for Executive Master's Students

Each project is designed to be completed in 25 minutes with minimal technical complexity.
UPDATED VERSION with improved model architectures and training parameters.
"""

# Import the main AI library - TensorFlow - this is like importing a toolbox for building AI
import tensorflow as tf
# Import NumPy for working with numbers and arrays (like Excel spreadsheets but for computers)
import numpy as np
# Import Pandas for working with data tables (like Excel but for AI)
import pandas as pd
# Import Matplotlib for creating charts and graphs
import matplotlib.pyplot as plt
# Import tools to split our data into training and testing sets
from sklearn.model_selection import train_test_split
# Import tools to standardize our data (make all numbers similar in scale)
from sklearn.preprocessing import StandardScaler
# Import warnings module to hide technical messages that might confuse students
import warnings
# Tell the computer to ignore warning messages so we can focus on the results
warnings.filterwarnings('ignore')

# Set a random seed - this is like setting a specific starting point so we get the same results every time
# Think of it like setting the same starting number in a random number generator
tf.random.set_seed(42)
np.random.seed(42)

# Print the version of TensorFlow we're using (like checking what version of software you have)
print("TensorFlow version:", tf.__version__)
# Print a welcome message to confirm everything is ready
print("Ready for Session 1 Projects!")

# ============================================================================
# PROJECT 1: Simple Number Prediction (25 minutes) 
# Business Context: Sales forecasting, demand prediction
# ============================================================================

def project_1_number_prediction():
    """
    PROJECT 1: Simple Number Prediction 
    Learning Goal: Understand how AI learns patterns
    Business Application: Predicting next month's sales based on historical data
    
    FIXES APPLIED:
    - Increased epochs from 100 to 500
    - Added second hidden layer (20 → 10 → 1 neurons)
    - Reduced learning rate to 0.01 for stable training
    - Added validation split to monitor training
    - Added early stopping to prevent overfitting
    """
    
    # Print a header with equal signs to make the output look organized
    print("\n" + "="*60)
    print("PROJECT 1: SIMPLE NUMBER PREDICTION")
    print("="*60)
    
    # Generate simple sequence data
    # Business analogy: Monthly sales data
    # Create an array of numbers from 1 to 12 (representing 12 months)
    months = np.arange(1, 13)  # 12 months
    # Create sales data that increases by $20K each month (linear growth pattern)
    # This simulates a business that's growing steadily
    sales = np.array([100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320])  # Linear growth
    
    # Display the historical sales data in a readable format
    print("Historical Sales Data:")
    # Loop through each month and its corresponding sales amount
    for month, sale in zip(months, sales):
        print(f"Month {month}: ${sale}K")
    
    # Prepare data for TensorFlow
    # Reshape the data - TensorFlow needs data in a specific format (like organizing files in folders)
    # reshape(-1, 1) means "make this into a column of numbers"
    X = months.reshape(-1, 1)  # Input: month number
    y = sales.reshape(-1, 1)   # Output: sales amount
    
    # Split data (use first 10 months for training, last 2 for testing)
    # This is like using 10 months of data to teach the AI, then testing it on the last 2 months
    # We do this to see how well the AI can predict new data it hasn't seen before
    X_train, X_test = X[:10], X[10:]  # First 10 months for training, last 2 for testing
    y_train, y_test = y[:10], y[10:]  # Corresponding sales data
    
    print(f"\nTraining on months 1-10, testing on months 11-12")
    
    # Build improved neural network with better architecture
    # This creates the AI "brain" - a series of connected layers that can learn patterns
    model = tf.keras.Sequential([
        # First layer: 20 neurons that look for patterns in the input (month number)
        # activation='relu' means the neurons only "fire" if they see something interesting
        # input_shape=(1,) means we're feeding in one number at a time (the month)
        tf.keras.layers.Dense(20, activation='relu', input_shape=(1,)),
        # Second layer: 10 neurons for better pattern recognition
        tf.keras.layers.Dense(10, activation='relu'),
        # Output layer: 1 neuron that gives us the final prediction (sales amount)
        # No activation function means it can output any number
        tf.keras.layers.Dense(1)
    ])
    
    # Configure how the AI learns with better learning rate
    # optimizer='adam' is like choosing the best learning strategy
    # learning_rate=0.01 means the AI learns at a good pace - not too fast, not too slow
    # loss='mse' means we measure errors by how far off our predictions are (squared)
    # metrics=['mae'] means we also track the average error in dollars
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
                  loss='mse', 
                  metrics=['mae'])
    
    # Add early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True
    )
    
    # Train the model
    # This is where the AI actually learns the pattern
    # epochs=500 means the AI looks at the data 500 times to learn the pattern
    # verbose=0 means don't show the detailed training progress (keep it simple)
    # validation_split=0.2 means use 20% of training data to check progress
    print("\nTraining the AI model...")
    history = model.fit(X_train, y_train, 
                       epochs=500, 
                       verbose=0,
                       validation_split=0.2,
                       callbacks=[early_stopping])
    
    # Make predictions
    # Now we test the AI on the months it hasn't seen before (months 11-12)
    predictions = model.predict(X_test)
    
    # Display the results
    print("\nResults:")
    # Show what the AI predicted vs what actually happened
    # predictions[0][0] gets the first prediction, predictions[1][0] gets the second
    print("Month 11 - Actual: $340K, Predicted: ${:.0f}K".format(predictions[0][0]))
    print("Month 12 - Actual: $360K, Predicted: ${:.0f}K".format(predictions[1][0]))
    
    # Calculate accuracy
    mae = np.mean(np.abs(predictions.flatten() - [340, 360]))
    print(f"Mean Absolute Error: ${mae:.0f}K")
    
    # Business insight - explain what this means in business terms
    print("\nBusiness Insight:")
    print("The AI learned the pattern: sales increase by $20K each month")
    print("This can help predict future sales for budgeting and planning")
    
    # Show the learning progress
    print(f"\nTraining completed in {len(history.history['loss'])} epochs")
    print(f"Final training loss: {history.history['loss'][-1]:.2f}")
    
    # Return the trained model and predictions so other parts of the program can use them
    return model, predictions

# ============================================================================
# PROJECT 2: Customer Satisfaction Predictor (25 minutes) - FIXED VERSION
# Business Context: Customer service, product quality
# ============================================================================

def project_2_customer_satisfaction():
    """
    PROJECT 2: Customer Satisfaction Predictor 
    Learning Goal: Binary classification (Yes/No predictions)
    Business Application: Identifying at-risk customers before they churn
    
    FIXES APPLIED:
    - Enhanced architecture with 3 hidden layers (32 → 16 → 8 → 1)
    - Reduced learning rate to 0.001 for better convergence
    - Added batch size specification
    - Improved dropout rate to 0.3
    - Increased epochs to 100
    """
    
    # Print header for this project
    print("\n" + "="*60)
    print("PROJECT 2: CUSTOMER SATISFACTION PREDICTOR")
    print("="*60)
    
    # Generate customer data
    # Set the random seed again to get consistent results
    np.random.seed(42)
    # Create 200 fake customers for our demonstration
    n_customers = 200
    
    # Customer features (normalized 0-1 scale)
    # We're creating fake customer data with different satisfaction factors
    # Each factor is scored from 0 (terrible) to 1 (excellent)
    data = {
        'response_time': np.random.uniform(0, 1, n_customers),  # 0=fast, 1=slow
        'product_quality': np.random.uniform(0, 1, n_customers),  # 0=poor, 1=excellent
        'price_satisfaction': np.random.uniform(0, 1, n_customers),  # 0=expensive, 1=good value
        'support_quality': np.random.uniform(0, 1, n_customers)  # 0=poor, 1=excellent
    }
    
    # Convert our data dictionary into a pandas DataFrame (like an Excel spreadsheet)
    df = pd.DataFrame(data)
    
    # Create satisfaction target (business logic)
    # Customers are satisfied if most factors are good
    # This is our business rule for determining if a customer is satisfied
    satisfaction_score = (
        (1 - df['response_time']) * 0.3 +  # Faster response = better (30% weight)
        df['product_quality'] * 0.3 +      # Higher quality = better (30% weight)
        df['price_satisfaction'] * 0.2 +   # Better value = better (20% weight)
        df['support_quality'] * 0.2        # Better support = better (20% weight)
    )
    
    # Convert to binary (satisfied = 1, not satisfied = 0)
    # If satisfaction score > 0.6, customer is satisfied (1), otherwise not satisfied (0)
    df['satisfied'] = (satisfaction_score > 0.6).astype(int)
    
    # Show a sample of our customer data
    print("Customer Data Sample:")
    print(df.head())  # Show first 5 rows
    print(f"\nSatisfaction Rate: {df['satisfied'].mean():.1%}")  # Show percentage of satisfied customers
    
    # Prepare data
    # Extract the features (input) and target (output) from our data
    # We use the 4 satisfaction factors to predict if customer is satisfied
    X = df[['response_time', 'product_quality', 'price_satisfaction', 'support_quality']].values
    y = df['satisfied'].values
    
    # Split data
    # Divide our data: 80% for training, 20% for testing
    # random_state=42 ensures we get the same split every time
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build improved classification model
    # Create a more complex AI brain for this classification task
    model = tf.keras.Sequential([
        # First layer: 32 neurons looking for patterns in 4 input features
        tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
        # Dropout layer: randomly turns off 30% of neurons during training to prevent overfitting
        # This is like making the AI more robust by not relying too much on any single pattern
        tf.keras.layers.Dropout(0.3),
        # Second layer: 16 neurons for more complex pattern recognition
        tf.keras.layers.Dense(16, activation='relu'),
        # Third layer: 8 neurons for refined patterns
        tf.keras.layers.Dense(8, activation='relu'),
        # Output layer: 1 neuron with sigmoid activation (outputs probability between 0 and 1)
        # Sigmoid is like a probability calculator - 0 = definitely not satisfied, 1 = definitely satisfied
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Configure the model for binary classification with better learning rate
    # binary_crossentropy is the right loss function for yes/no predictions
    # accuracy measures how often the AI gets the right answer
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True
    )
    
    # Train model
    # Teach the AI to recognize patterns that indicate customer satisfaction
    print("\nTraining customer satisfaction predictor...")
    # validation_split=0.2 means use 20% of training data to check progress during training
    history = model.fit(X_train, y_train, 
                       epochs=100, 
                       validation_split=0.2, 
                       verbose=0,
                       batch_size=32,
                       callbacks=[early_stopping])
    
    # Evaluate model
    # Test how well the AI performs on data it hasn't seen before
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nModel Accuracy: {test_accuracy:.1%}")
    print(f"Training Accuracy: {history.history['accuracy'][-1]:.1%}")
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.1%}")
    
    # Make predictions on new customers
    # Test the AI on 3 new customers with different characteristics
    new_customers = np.array([
        [0.8, 0.9, 0.7, 0.8],  # Good customer (slow response but good quality, value, support)
        [0.2, 0.3, 0.4, 0.2],  # At-risk customer (fast response but poor quality, value, support)
        [0.5, 0.6, 0.5, 0.5]   # Average customer (middle-of-the-road on all factors)
    ])
    
    # Get predictions for these new customers
    predictions = model.predict(new_customers)
    
    # Display the results
    print("\nPredictions for New Customers:")
    customer_types = ["Good Customer", "At-Risk Customer", "Average Customer"]
    # Loop through each customer and their prediction
    for i, (customer_type, pred) in enumerate(zip(customer_types, predictions)):
        satisfaction_prob = pred[0]  # Get the probability of satisfaction
        print(f"{customer_type}: {satisfaction_prob:.1%} chance of satisfaction")
    
    # Business insight
    print("\nBusiness Insight:")
    print("Use this model to identify customers at risk of churning")
    print("Focus improvement efforts on customers with low satisfaction scores")
    
    # Show training progress
    print(f"\nTraining completed in {len(history.history['loss'])} epochs")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    
    return model, predictions

# ============================================================================
# PROJECT 3: Price Optimization Model (25 minutes) 
# Business Context: Pricing strategy, revenue optimization
# ============================================================================

def project_3_price_optimization():
    """
    PROJECT 3: Price Optimization Model
    Learning Goal: Regression (predicting continuous values)
    Business Application: Dynamic pricing for e-commerce or services
    
    FIXES APPLIED:
    - Deeper network with 4 layers (64 → 32 → 16 → 1)
    - Better learning rate of 0.001
    - Increased epochs to 200
    - Added batch processing for stability
    - Improved dropout rate to 0.3
    """
    
    # Print header for this project
    print("\n" + "="*60)
    print("PROJECT 3: PRICE OPTIMIZATION MODEL")
    print("="*60)
    
    # Generate pricing data
    # Set random seed for consistent results
    np.random.seed(42)
    # Create 150 fake products for our pricing analysis
    n_products = 150
    
    # Product features that influence optimal pricing
    data = {
        'competitor_price': np.random.uniform(50, 200, n_products),  # What competitors charge
        'demand_level': np.random.uniform(0.3, 1.0, n_products),  # 0.3=low demand, 1.0=high demand
        'seasonality': np.random.uniform(0.5, 1.5, n_products),   # 0.5=off-season, 1.5=peak season
        'product_quality': np.random.uniform(0.6, 1.0, n_products)  # 0.6=basic quality, 1.0=premium
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Calculate optimal price (business logic)
    # This is our pricing strategy formula
    base_price = df['competitor_price'] * 0.9  # Start 10% below competitor
    demand_multiplier = 1 + (df['demand_level'] - 0.5) * 0.4  # Higher demand = higher price
    season_multiplier = df['seasonality']  # Peak season = higher price
    quality_multiplier = 1 + (df['product_quality'] - 0.8) * 0.5  # Higher quality = higher price
    
    # Calculate the final optimal price by multiplying all factors
    df['optimal_price'] = base_price * demand_multiplier * season_multiplier * quality_multiplier
    
    # Show sample of our pricing data
    print("Product Pricing Data Sample:")
    print(df.head())
    print(f"\nPrice Range: ${df['optimal_price'].min():.2f} - ${df['optimal_price'].max():.2f}")
    
    # Prepare data
    # Use the 4 factors to predict the optimal price
    X = df[['competitor_price', 'demand_level', 'seasonality', 'product_quality']].values
    y = df['optimal_price'].values
    
    # Split data
    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    # Standardize the data so all numbers are on the same scale (like converting inches to centimeters)
    # This helps the AI learn more effectively
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Learn the scaling and apply to training data
    X_test_scaled = scaler.transform(X_test)        # Apply the same scaling to test data
    
    # Build improved regression model
    # Create AI brain for predicting continuous price values
    model = tf.keras.Sequential([
        # First layer: 64 neurons for complex pattern recognition
        tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
        # Dropout to prevent overfitting
        tf.keras.layers.Dropout(0.3),
        # Second layer: 32 neurons for refined pattern recognition
        tf.keras.layers.Dense(32, activation='relu'),
        # Third layer: 16 neurons for more refined patterns
        tf.keras.layers.Dense(16, activation='relu'),
        # Output layer: 1 neuron for the price prediction (no activation = can output any number)
        tf.keras.layers.Dense(1)
    ])
    
    # Configure for regression (predicting continuous values) with better learning rate
    # mse = mean squared error (measures how far off our price predictions are)
    # mae = mean absolute error (average dollar amount we're off by)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='mse', 
                  metrics=['mae'])
    
    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_mae',
        patience=30,
        restore_best_weights=True
    )
    
    # Train model
    # Teach the AI to predict optimal prices based on market factors
    print("\nTraining price optimization model...")
    history = model.fit(X_train_scaled, y_train, 
                       epochs=200, 
                       validation_split=0.2, 
                       verbose=0,
                       batch_size=32,
                       callbacks=[early_stopping])
    
    # Evaluate model
    # Test how well the AI predicts prices
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nModel Performance:")
    print(f"Mean Absolute Error: ${test_mae:.2f}")  # Average dollar amount we're off by
    print(f"Training MAE: ${history.history['mae'][-1]:.2f}")
    print(f"Validation MAE: ${history.history['val_mae'][-1]:.2f}")
    
    # Make pricing recommendations
    # Test the AI on 3 new product scenarios
    new_products = np.array([
        [100, 0.8, 1.2, 0.9],  # High-demand, peak season, good quality
        [150, 0.4, 0.8, 0.7],  # Low-demand, off-season, basic quality
        [80, 0.9, 1.0, 0.95]   # Very high demand, normal season, premium quality
    ])
    
    # Scale the new products using the same scaling we learned from training data
    new_products_scaled = scaler.transform(new_products)
    # Get price predictions
    price_predictions = model.predict(new_products_scaled)
    
    # Display the pricing recommendations
    print("\nPricing Recommendations:")
    scenarios = ["High-Demand Premium", "Low-Demand Basic", "Very High-Demand Premium"]
    for i, (scenario, pred_price) in enumerate(zip(scenarios, price_predictions)):
        print(f"{scenario}: Recommended price ${pred_price[0]:.2f}")
    
    # Business insight
    print("\nBusiness Insight:")
    print("Use this model to set optimal prices based on market conditions")
    print("Consider competitor prices, demand, seasonality, and product quality")
    
    # Show training progress
    print(f"\nTraining completed in {len(history.history['loss'])} epochs")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    
    return model, price_predictions

# ============================================================================
# CLASSWORK: Business Problem Identification (15 minutes)
# ============================================================================

def classwork_business_problems():
    """
    CLASSWORK: Business Problem Identification
    Activity: Students identify 3 AI opportunities in their industry
    Deliverable: One-page summary of potential TensorFlow applications
    """
    
    # Print header for classwork
    print("\n" + "="*60)
    print("CLASSWORK: BUSINESS PROBLEM IDENTIFICATION")
    print("="*60)
    
    # Instructions to students
    print("Instructions:")
    print("1. Think about your industry/company")
    print("2. Identify 3 business problems that could be solved with AI")
    print("3. For each problem, specify:")
    print("   - Business impact (revenue, cost, efficiency)")
    print("   - Data requirements")
    print("   - Implementation challenges")
    print("   - Expected ROI")
    
    # Template to help students structure their thinking
    print("\nExample Template:")
    print("Problem 1: [Describe the business problem]")
    print("Impact: [How it affects the business]")
    print("Data: [What data is needed]")
    print("Challenges: [Implementation difficulties]")
    print("ROI: [Expected return on investment]")
    
    # Remind students of the concepts they learned today
    print("\nUse the concepts learned today:")
    print("- Pattern recognition (Project 1)")
    print("- Classification (Project 2)")
    print("- Regression (Project 3)")
   

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# This section runs when the file is executed directly (not imported)
if __name__ == "__main__":
    # Print session header
    print("SESSION 1: TensorFlow Fundamentals & Basic Neural Networks")
    print("Duration: 2 hours | 3 Projects + 1 Classwork")
    print("UPDATED VERSION with improved model architectures and training parameters")
    
    # Run all projects
    print("\nStarting projects...")
    
    # Project 1: Number Prediction
    # Call the function and store the results
    model1, pred1 = project_1_number_prediction()
    
    # Project 2: Customer Satisfaction
    # Call the function and store the results
    model2, pred2 = project_2_customer_satisfaction()
    
    # Project 3: Price Optimization
    # Call the function and store the results
    model3, pred3 = project_3_price_optimization()
    
    # Classwork
    # Run the classwork activity
    classwork_business_problems()
    
    # Print session completion message
    print("\n" + "="*60)
    print("SESSION 1 COMPLETE!")
    print("="*60)
    print("Key Learnings:")
    print("1. AI can learn patterns from data")
    print("2. Classification helps make yes/no decisions")
    print("3. Regression predicts continuous values")
    print("4. All concepts have direct business applications")
    print("\nKey Improvements Made:")
    print("- Better model architectures with more layers and neurons")
    print("- Optimized learning rates for stable training")
    print("- Increased training epochs for better pattern learning")
    print("- Added validation monitoring to prevent overfitting")
    print("- Improved data scaling for better model performance")
    print("\nNext Session: Computer Vision for Business") 