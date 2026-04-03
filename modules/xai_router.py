import shap

def route_explanation(model, X_instance, risk_score, modality_type):
    """
    Dynamic XAI router (Novelty 3).
    Selects the optimal explanation method based on clinical urgency and modality.
    """
    if risk_score > 0.75:
        # High urgency - requires fast explanation
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_instance)
        return {"method": "SHAP_fast", "values": shap_vals, "explainer": explainer}
        
    elif modality_type == "imaging":
        # Specific fallback for high-density modalities
        return {"method": "feature_importance", "values": model.feature_importances_}
        
    else:
        # Low urgency - can afford full detail
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X_instance)
        return {"method": "SHAP_waterfall", "values": explanation}
