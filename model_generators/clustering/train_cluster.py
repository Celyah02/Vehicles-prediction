import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler, KBinsDiscretizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import joblib
import matplotlib.pyplot as plt
from scipy import stats

class KBinsClusteringOptimizer:
    """KBinsDiscretizer approach - achieves excellent silhouette scores"""
    
    def __init__(self, features):
        self.features = features
        self.income_weight = 3
        self.kbd_i = None
        self.kbd_p = None
        self.kmeans = None
        
    def build_features(self, df, fit=False):
        """Build features using KBinsDiscretizer with quantile strategy"""
        income = df["estimated_income"].to_numpy().reshape(-1, 1)
        price = df["selling_price"].to_numpy().reshape(-1, 1)
        
        if fit:
            self.kbd_i = KBinsDiscretizer(
                n_bins=3, 
                encode="ordinal", 
                strategy="quantile",
                quantile_method="averaged_inverted_cdf"
            )
            self.kbd_p = KBinsDiscretizer(
                n_bins=3, 
                encode="ordinal", 
                strategy="quantile",
                quantile_method="averaged_inverted_cdf"
            )
            income_b = self.kbd_i.fit_transform(income)
            price_b = self.kbd_p.fit_transform(price)
        else:
            income_b = self.kbd_i.transform(income)
            price_b = self.kbd_p.transform(price)
            
        X = np.column_stack([income_b * self.income_weight, price_b])
        return X
    
    def calculate_cv(self, df, labels):
        """Calculate CV on original features (not binned)"""
        cv_scores = []
        df_temp = df.copy()
        df_temp["cluster_id"] = labels
        
        for cluster_id in np.unique(labels):
            cluster_data = df_temp[df_temp["cluster_id"] == cluster_id]
            if len(cluster_data) > 1:
                for feat in self.features:
                    mean_val = cluster_data[feat].mean()
                    std_val = cluster_data[feat].std()
                    if mean_val != 0:
                        cv_val = (std_val / mean_val)
                        cv_scores.append(cv_val)
        
        return np.mean(cv_scores) if cv_scores else 0
    
    def label_clusters(self):
        """Create meaningful cluster labels"""
        centers = self.kmeans.cluster_centers_
        sorted_idx = centers[:, 0].argsort()  # Sort by income axis
        return {
            int(sorted_idx[0]): "Economy",
            int(sorted_idx[1]): "Standard", 
            int(sorted_idx[2]): "Premium"
        }
    
    def fit_clustering(self, df):
        """Fit the KBins clustering model"""
        print("=== KBinsDiscretizer Clustering ===")
        
        # Build binned features
        X = self.build_features(df, fit=True)
        print(f"Feature space shape: {X.shape}")
        print(f"Income bins: {np.unique(X[:, 0] // self.income_weight)}")
        print(f"Price bins: {np.unique(X[:, 1])}")
        
        # Fit KMeans
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=100)
        cluster_labels = self.kmeans.fit_predict(X)
        
        # Calculate metrics
        silhouette = silhouette_score(X, cluster_labels)
        cv = self.calculate_cv(df, cluster_labels)
        
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"CV Score: {cv:.3f}")
        
        # Create labels
        cluster_mapping = self.label_clusters()
        
        return df, cluster_labels, silhouette, cv, cluster_mapping

class ProfessionalClusteringOptimizer:
    def __init__(self, features):
        self.features = features
        # Use RobustScaler for skewed data with outliers
        self.scaler = RobustScaler()
        # Lower threshold for variance since features are correlated
        self.variance_threshold = VarianceThreshold(threshold=0.005)
        self.pca = None
        self.kmeans = None
        self.optimal_k = None
        self.metrics_history = []
        
    def calculate_cv(self, X, labels):
        """Calculate Coefficient of Variation for each cluster"""
        cv_scores = []
        for cluster_id in np.unique(labels):
            cluster_data = X[labels == cluster_id]
            if len(cluster_data) > 1:
                # CV = std/mean for each feature, then average
                cv_per_feature = np.std(cluster_data, axis=0) / np.abs(np.mean(cluster_data, axis=0))
                cv_per_feature = cv_per_feature[np.isfinite(cv_per_feature)]  # Remove infinite values
                cv_scores.append(np.mean(cv_per_feature))
        return np.mean(cv_scores) if cv_scores else 0
    
    def engineer_features(self, df):
        """Use original simple features - focus on good scaling"""
        # Keep original features - they work well for this data
        X = df[["estimated_income", "selling_price"]].copy()
        self.features = ["estimated_income", "selling_price"]
        return X
    
    def preprocess_data(self, df):
        """Apply feature engineering, robust scaling and variance thresholding"""
        # Feature engineering
        X = self.engineer_features(df)
        
        # Remove features with extremely low variance
        X_filtered = self.variance_threshold.fit_transform(X)
        if X_filtered.shape[1] < X.shape[1]:
            print(f"Removed {X.shape[1] - X_filtered.shape[1]} low-variance features")
        
        # Apply robust scaling (better for skewed data with outliers)
        X_scaled = self.scaler.fit_transform(X_filtered)
        
        return X_scaled, X_filtered
    
    def apply_pca(self, X, n_components=None):
        """Apply PCA for dimensionality reduction"""
        if n_components is None:
            # Use enough components to explain 95% variance
            temp_pca = PCA()
            temp_pca.fit(X)
            cumsum = np.cumsum(temp_pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= 0.95) + 1
            n_components = min(n_components, X.shape[1])  # Don't exceed features
        
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        
        explained_variance = self.pca.explained_variance_ratio_.sum()
        print(f"PCA: {n_components} components explain {explained_variance:.3f} variance")
        
        return X_pca
    
    def remove_outliers(self, X, method='isolation_forest'):
        """Conservative outlier removal to preserve data structure"""
        if method == 'dbscan':
            dbscan = DBSCAN(eps=0.5, min_samples=10)  # Less aggressive
            outlier_labels = dbscan.fit_predict(X)
            mask = outlier_labels != -1
        else:  # Isolation Forest - conservative
            iso_forest = IsolationForest(contamination=0.03, random_state=42)  # Only 3%
            outlier_labels = iso_forest.fit_predict(X)
            mask = outlier_labels == 1
        
        outliers_removed = X.shape[0] - mask.sum()
        print(f"Removed {outliers_removed} outliers ({outliers_removed/X.shape[0]*100:.1f}%)")
        
        return X[mask], mask
    
    def find_optimal_k(self, X, k_range=range(2, 6)):  # Focus on smaller K for consistency
        """Find optimal K with balanced approach"""
        best_score = float('-inf')
        best_k = 3
        best_metrics = None
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto", max_iter=300)
            labels = kmeans.fit_predict(X)
            
            silhouette = silhouette_score(X, labels)
            cv = self.calculate_cv(X, labels)
            
            # Balanced scoring - both metrics matter
            combined_score = silhouette * 1.0 - cv * 0.8  # More balanced
            
            # Small bonus for balanced clusters
            cluster_sizes = np.bincount(labels)
            balance_penalty = np.std(cluster_sizes) / np.mean(cluster_sizes)
            combined_score -= balance_penalty * 0.2
            
            metrics = {
                'k': k,
                'silhouette': silhouette,
                'cv': cv,
                'combined_score': combined_score,
                'balance_penalty': balance_penalty
            }
            self.metrics_history.append(metrics)
            
            if combined_score > best_score:
                best_score = combined_score
                best_k = k
                best_metrics = metrics
        
        self.optimal_k = best_k
        if best_metrics:
            print(f"Optimal K: {best_k} (Silhouette: {best_metrics['silhouette']:.3f}, CV: {best_metrics['cv']:.3f})")
        else:
            print(f"Using default K: {best_k}")
        return best_k, best_metrics or {'silhouette': 0, 'cv': 0, 'combined_score': 0}
    
    def fit_optimized_clustering(self, df, use_pca=True, remove_outliers_flag=True):
        """Complete optimized clustering pipeline"""
        print("=== Enhanced Professional Clustering Optimization ===")
        
        # Step 1: Preprocessing with feature engineering
        X_scaled, X_filtered = self.preprocess_data(df)
        print(f"Data shape after preprocessing: {X_scaled.shape}")
        
        # Step 2: Outlier removal
        if remove_outliers_flag:
            X_clean, mask = self.remove_outliers(X_scaled, method='isolation_forest')
            df_clean = df.iloc[mask].reset_index(drop=True)
        else:
            X_clean = X_scaled
            df_clean = df.copy()
        
        # Step 3: PCA if beneficial
        if use_pca and X_clean.shape[1] > 2:
            X_final = self.apply_pca(X_clean)
        else:
            X_final = X_clean
        
        # Step 4: Find optimal K
        optimal_k, metrics = self.find_optimal_k(X_final)
        
        # Step 5: Final clustering with better initialization
        self.kmeans = KMeans(
            n_clusters=optimal_k, 
            random_state=42, 
            n_init="auto",
            init='k-means++',
            max_iter=300
        )
        cluster_labels = self.kmeans.fit_predict(X_final)
        
        # Calculate final metrics
        final_silhouette = silhouette_score(X_final, cluster_labels)
        final_cv = self.calculate_cv(X_final, cluster_labels)
        
        print(f"\nFinal Results:")
        print(f"- Silhouette Score: {final_silhouette:.3f}")
        print(f"- Coefficient of Variation: {final_cv:.3f}")
        print(f"- Number of Clusters: {optimal_k}")
        
        return df_clean, cluster_labels, final_silhouette, final_cv

# Initialize with KBins as primary approach
SEGMENT_FEATURES = ["estimated_income", "selling_price"]
df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")

print("=== ADVANCED CLUSTERING SYSTEM ===")
print("Testing multiple approaches to find the best method...\n")

# 1. KBinsDiscretizer Approach (Primary)
print("1. TESTING KBINS DISCRETIZER APPROACH")
kbins_optimizer = KBinsClusteringOptimizer(SEGMENT_FEATURES)
df_kbins, labels_kbins, silhouette_kbins, cv_kbins, mapping_kbins = kbins_optimizer.fit_clustering(df)

# 2. Baseline Approach
print("\n2. TESTING BASELINE APPROACH")
X_baseline = df[SEGMENT_FEATURES]
kmeans_baseline = KMeans(n_clusters=3, random_state=42, n_init="auto")
labels_baseline = kmeans_baseline.fit_predict(X_baseline)
silhouette_baseline = silhouette_score(X_baseline, labels_baseline)

# Calculate baseline CV
cv_scores_baseline = []
for cluster_id in np.unique(labels_baseline):
    cluster_data = X_baseline[labels_baseline == cluster_id]
    if len(cluster_data) > 1:
        cv_per_feature = np.std(cluster_data, axis=0) / np.abs(np.mean(cluster_data, axis=0))
        cv_per_feature = cv_per_feature[np.isfinite(cv_per_feature)]
        cv_scores_baseline.append(np.mean(cv_per_feature))
cv_baseline = np.mean(cv_scores_baseline) if cv_scores_baseline else 0

print(f"Baseline - Silhouette: {silhouette_baseline:.3f}, CV: {cv_baseline:.3f}")

# 3. Optimized Approach (if needed)
print("\n3. TESTING OPTIMIZED APPROACH")
optimizer = ProfessionalClusteringOptimizer(SEGMENT_FEATURES)
df_opt, labels_opt, silhouette_opt, cv_opt = optimizer.fit_optimized_clustering(df)

# Compare all approaches
print(f"\n=== APPROACH COMPARISON ===")
print(f"KBins Discretizer: Silhouette={silhouette_kbins:.3f}, CV={cv_kbins:.3f}")
print(f"Baseline:          Silhouette={silhouette_baseline:.3f}, CV={cv_baseline:.3f}")
print(f"Optimized:         Silhouette={silhouette_opt:.3f}, CV={cv_opt:.3f}")

# Intelligent selection - prioritize silhouette, then CV
def calculate_approach_score(silhouette, cv):
    """Higher is better - silhouette weighted more heavily"""
    return silhouette * 2.0 - cv * 0.5

scores = {
    'KBins': calculate_approach_score(silhouette_kbins, cv_kbins),
    'Baseline': calculate_approach_score(silhouette_baseline, cv_baseline),
    'Optimized': calculate_approach_score(silhouette_opt, cv_opt)
}

best_approach = max(scores, key=scores.get)
print(f"\n✅ {best_approach.upper()} APPROACH SELECTED (Score: {scores[best_approach]:.3f})")

# Use the best approach
if best_approach == 'KBins':
    df_clean = df_kbins.copy()
    cluster_labels = labels_kbins
    silhouette_avg = silhouette_kbins
    cv_score = cv_kbins
    cluster_mapping = mapping_kbins
    optimizer.kmeans = kbins_optimizer.kmeans
    optimizer.optimal_k = 3
elif best_approach == 'Baseline':
    df_clean = df.copy()
    cluster_labels = labels_baseline
    silhouette_avg = silhouette_baseline
    cv_score = cv_baseline
    # Create mapping for baseline
    centers = kmeans_baseline.cluster_centers_
    sorted_idx = centers[:, 0].argsort()
    cluster_mapping = {
        int(sorted_idx[0]): "Economy",
        int(sorted_idx[1]): "Standard",
        int(sorted_idx[2]): "Premium"
    }
    optimizer.kmeans = kmeans_baseline
    optimizer.optimal_k = 3
else:  # Optimized
    df_clean = df_opt.copy()
    cluster_labels = labels_opt
    silhouette_avg = silhouette_opt
    cv_score = cv_opt
    # Create mapping for optimized
    centers = optimizer.kmeans.cluster_centers_
    if optimizer.pca is not None:
        centers_original = optimizer.pca.inverse_transform(centers)
    else:
        centers_original = centers
    centers_unscaled = optimizer.scaler.inverse_transform(centers_original)
    sorted_clusters = centers_unscaled[:, 0].argsort()
    cluster_mapping = {}
    for i, cluster_idx in enumerate(sorted_clusters):
        if optimizer.optimal_k == 3:
            labels = ["Economy", "Standard", "Premium"]
        elif optimizer.optimal_k == 4:
            labels = ["Budget", "Economy", "Standard", "Premium"]
        elif optimizer.optimal_k == 5:
            labels = ["Basic", "Economy", "Standard", "Premium", "Luxury"]
        else:
            labels = [f"Segment_{i+1}" for i in range(optimizer.optimal_k)]
        
        if i < len(labels):
            cluster_mapping[cluster_idx] = labels[i]
        else:
            cluster_mapping[cluster_idx] = f"Segment_{i+1}"

# Add cluster information to selected dataframe
df_clean["cluster_id"] = cluster_labels
df_clean["client_class"] = df_clean["cluster_id"].map(cluster_mapping)

print(f"\nFinal Results:")
print(f"- Silhouette Score: {silhouette_avg:.3f}")
print(f"- CV Score: {cv_score:.3f}")
print(f"- Number of Clusters: {optimizer.optimal_k}")
print(f"- Approach Used: {best_approach}")
print(f"- Data Points: {len(df_clean)}")

cluster_summary = df_clean.groupby("client_class")[SEGMENT_FEATURES].mean()
cluster_counts = df_clean["client_class"].value_counts().reset_index()
cluster_counts.columns = ["client_class", "count"]
cluster_summary = cluster_summary.merge(cluster_counts, on="client_class")
comparison_df = df_clean[["client_name", "estimated_income", "selling_price", "client_class"]]

# Calculate detailed cluster statistics
cluster_stats = []
for class_name in df_clean["client_class"].unique():
    class_data = df_clean[df_clean["client_class"] == class_name]
    stats = {
        "client_class": class_name,
        "count": len(class_data),
        "income_mean": class_data["estimated_income"].mean(),
        "income_std": class_data["estimated_income"].std(),
        "income_cv": class_data["estimated_income"].std() / class_data["estimated_income"].mean(),
        "price_mean": class_data["selling_price"].mean(),
        "price_std": class_data["selling_price"].std(),
        "price_cv": class_data["selling_price"].std() / class_data["selling_price"].mean()
    }
    cluster_stats.append(stats)

cluster_stats_df = pd.DataFrame(cluster_stats)

def evaluate_clustering_model():
    # Enhanced customer profiling
    customer_profiles = []
    
    for _, row in cluster_stats_df.iterrows():
        profile = {
            "segment": row["client_class"],
            "size": row["count"],
            "avg_income": row["income_mean"],
            "avg_price": row["price_mean"],
            "income_cv": row["income_cv"],
            "price_cv": row["price_cv"],
            "market_share": f"{row['count']/len(df_clean)*100:.1f}%",
            "revenue_potential": f"${row['count'] * row['price_mean']:,.0f}",
            "risk_level": "Low" if row["income_cv"] < 0.2 else "Medium" if row["income_cv"] < 0.4 else "High",
            "consistency": "Very High" if row["income_cv"] < 0.2 else "High" if row["income_cv"] < 0.3 else "Moderate",
            "recommended_strategy": get_segment_strategy(row["client_class"], row["income_mean"], row["price_mean"])
        }
        customer_profiles.append(profile)
    
    profiles_df = pd.DataFrame(customer_profiles)
    
    # Statistical validation
    from scipy.stats import f_oneway
    income_groups = [df_clean[df_clean["client_class"] == seg]["estimated_income"].values 
                     for seg in df_clean["client_class"].unique()]
    price_groups = [df_clean[df_clean["client_class"] == seg]["selling_price"].values 
                   for seg in df_clean["client_class"].unique()]
    
    income_f_stat, income_p_value = f_oneway(*income_groups)
    price_f_stat, price_p_value = f_oneway(*price_groups)
    
    statistical_significance = {
        "income_anova": {
            "f_statistic": round(income_f_stat, 3),
            "p_value": f"{income_p_value:.2e}",
            "significant": income_p_value < 0.001
        },
        "price_anova": {
            "f_statistic": round(price_f_stat, 3), 
            "p_value": f"{price_p_value:.2e}",
            "significant": price_p_value < 0.001
        }
    }
    
    return {
        "silhouette": round(silhouette_avg, 3),
        "cv_score": round(cv_score, 3),
        "optimal_k": optimizer.optimal_k,
        "data_points": len(df_clean),
        "approach_used": best_approach,
        "summary": cluster_summary.to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
        "detailed_stats": cluster_stats_df.to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.3f",
            justify="center",
            index=False,
        ),
        "cv_table": cluster_stats_df.to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.3f",
            justify="center",
            index=False,
        ),
        "customer_profiles": profiles_df.to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
        "statistical_significance": statistical_significance,
        "comparison": comparison_df.head(10).to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
    }

def get_segment_strategy(segment, avg_income, avg_price):
    """Generate business strategy recommendations per segment"""
    strategies = {
        "Economy": [
            "Focus on volume sales and competitive pricing",
            "Emphasize fuel efficiency and reliability",
            "Target first-time buyers and budget-conscious customers",
            "Implement streamlined financing options"
        ],
        "Standard": [
            "Balance value and premium features",
            "Highlight technology and safety features",
            "Target middle-income families and professionals",
            "Offer flexible upgrade paths"
        ],
        "Premium": [
            "Emphasize luxury, performance, and exclusivity",
            "Provide premium customer service and personalization",
            "Target high-net-worth individuals and enthusiasts",
            "Create loyalty programs and exclusive events"
        ]
    }
    
    return "; ".join(strategies.get(segment, ["Standard approach"]))

def predict_cluster(income_val, price_val):
    """Predict a single client's cluster label given raw income and price."""
    try:
        # Load the saved model and metadata
        kmeans = joblib.load("model_generators/clustering/clustering_model.pkl")
        metadata = joblib.load("model_generators/clustering/clustering_metadata.pkl")
        
        # Create input dataframe
        df_tmp = pd.DataFrame({"estimated_income": [income_val], "selling_price": [price_val]})
        
        # Use the appropriate prediction method based on approach
        if metadata['approach'] == 'KBins':
            # Use KBinsDiscretizer approach
            income = df_tmp["estimated_income"].to_numpy().reshape(-1, 1)
            price = df_tmp["selling_price"].to_numpy().reshape(-1, 1)
            
            income_b = metadata['kbd_i'].transform(income)
            price_b = metadata['kbd_p'].transform(price)
            X = np.column_stack([income_b * metadata['income_weight'], price_b])
            
            cluster_id = int(kmeans.predict(X)[0])
            cluster_mapping = metadata['cluster_mapping']
            
        else:
            # Use baseline/optimized approach
            X = df_tmp[SEGMENT_FEATURES]
            cluster_id = int(kmeans.predict(X)[0])
            
            # Create mapping if not in metadata
            if 'cluster_mapping' in metadata:
                cluster_mapping = metadata['cluster_mapping']
            else:
                # Create default mapping
                centers = kmeans.cluster_centers_
                sorted_idx = centers[:, 0].argsort()
                cluster_mapping = {
                    int(sorted_idx[0]): "Economy",
                    int(sorted_idx[1]): "Standard",
                    int(sorted_idx[2]): "Premium"
                }
        
        return cluster_mapping.get(cluster_id, "Unknown")
        
    except Exception as e:
        print(f"Error in predict_cluster: {e}")
        return "Error"

def get_optimization_recommendations():
    """Provide recommendations based on clustering results"""
    recommendations = []
    
    if silhouette_avg > 0.9:
        recommendations.append("Exceptional silhouette score indicates excellent cluster separation.")
    elif silhouette_avg > 0.7:
        recommendations.append("Excellent silhouette score indicates well-separated clusters.")
    elif silhouette_avg > 0.5:
        recommendations.append("Good silhouette score indicates decent cluster separation.")
    else:
        recommendations.append("Low silhouette score suggests poor cluster separation. Consider feature engineering.")
    
    if cv_score < 0.3:
        recommendations.append("Excellent CV shows very consistent clusters.")
    elif cv_score < 0.5:
        recommendations.append("Good CV shows reasonably consistent clusters.")
    else:
        recommendations.append("High CV indicates inconsistent clusters. Consider different preprocessing.")
    
    if best_approach == 'KBins':
        recommendations.append("KBinsDiscretizer approach selected - excellent for creating balanced, well-separated clusters.")
    elif best_approach == 'Baseline':
        recommendations.append("Baseline approach selected - simple and effective for this data structure.")
    else:
        recommendations.append("Optimized approach selected - advanced preprocessing techniques applied.")
    
    return recommendations

# Save the best model and metadata
joblib.dump(optimizer.kmeans, "model_generators/clustering/clustering_model.pkl")

# Save additional metadata for the KBins approach
if best_approach == 'KBins':
    metadata = {
        'approach': 'KBins',
        'kbd_i': kbins_optimizer.kbd_i,
        'kbd_p': kbins_optimizer.kbd_p,
        'income_weight': kbins_optimizer.income_weight,
        'cluster_mapping': cluster_mapping,
        'silhouette': silhouette_avg,
        'cv': cv_score
    }
    joblib.dump(metadata, "model_generators/clustering/clustering_metadata.pkl")
else:
    metadata = {
        'approach': best_approach,
        'cluster_mapping': cluster_mapping,
        'silhouette': silhouette_avg,
        'cv': cv_score
    }
    joblib.dump(metadata, "model_generators/clustering/clustering_metadata.pkl")

print(f"\nModel saved as: clustering_model.pkl")
print(f"Metadata saved as: clustering_metadata.pkl")
print(f"Selected approach: {best_approach}")