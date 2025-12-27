"""
# ------------------------------------------------------- #
# SAMPLE DATA GENERATOR                                   #
# ------------------------------------------------------- #
"""

import pandas as pd
from datetime import datetime
from typing import Optional

def generate_sample_data(date: Optional[str] = None) -> pd.DataFrame:
    """Generate sample data for testing."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    return pd.DataFrame([
        
        {
            'date': date,
            'ad_name': 'P_Spring_StackedOffer_Video1',
            'campaign_name': 'TOF - Broad Audiences',
            'frequency_7d': 1.8,
            'landing_page_url': 'https://example.com/greenhouses',
            'bounce_rate': 78,
            'avg_session_duration_seconds': 65,
            'spend': 250,
            'atc': 6,
            'ic': 2,
            'orders': 0,
            'revenue': 0
        },
        {
            'date': date,
            'ad_name': 'P_Premium_Carousel_v2',
            'campaign_name': 'Prospecting - Interest',
            'frequency_7d': 2.1,
            'landing_page_url': 'https://example.com/premium',
            'bounce_rate': 62,
            'avg_session_duration_seconds': 115,
            'spend': 180,
            'atc': 12,
            'ic': 5,
            'orders': 1,
            'revenue': 2950
        },
        {
            'date': date,
            'ad_name': 'Cold_UGC_Testimonial',
            'campaign_name': 'TOF - Lookalike',
            'frequency_7d': 1.5,
            'landing_page_url': 'https://example.com/reviews',
            'bounce_rate': 58,
            'avg_session_duration_seconds': 125,
            'spend': 320,
            'atc': 15,
            'ic': 8,
            'orders': 2,
            'revenue': 6200
        },
        
        {
            'date': date,
            'ad_name': 'R_CartAbandon_Urgency',
            'campaign_name': 'Retargeting - Cart',
            'frequency_7d': 3.8,
            'landing_page_url': 'https://example.com/cart',
            'bounce_rate': 45,
            'avg_session_duration_seconds': 135,
            'spend': 150,
            'atc': 18,
            'ic': 12,
            'orders': 4,
            'revenue': 12400
        },
        {
            'date': date,
            'ad_name': 'BOF_FinalOffer_Static',
            'campaign_name': 'Remarketing - Visitors',
            'frequency_7d': 4.2,
            'landing_page_url': 'https://example.com/offer',
            'bounce_rate': 68,
            'avg_session_duration_seconds': 82,
            'spend': 200,
            'atc': 8,
            'ic': 4,
            'orders': 1,
            'revenue': 2800
        },
        {
            'date': date,
            'ad_name': 'MOF_FeatureHighlight',
            'campaign_name': 'Retargeting - Engagers',
            'frequency_7d': 3.2,
            'landing_page_url': 'https://example.com/features',
            'bounce_rate': 52,
            'avg_session_duration_seconds': 142,
            'spend': 120,
            'atc': 10,
            'ic': 6,
            'orders': 2,
            'revenue': 5900
        },
        
        {
            'date': date,
            'ad_name': 'SpringSale_Video_v3',
            'campaign_name': 'Campaign_Seasonal',
            'frequency_7d': 2.0,
            'landing_page_url': 'https://example.com/sale',
            'bounce_rate': 70,
            'avg_session_duration_seconds': 95,
            'spend': 100,
            'atc': 5,
            'ic': 2,
            'orders': 0,
            'revenue': 0
        },
        {
            'date': date,
            'ad_name': 'BundleDeal_Carousel',
            'campaign_name': 'Campaign_Promo',
            'frequency_7d': 3.5,
            'landing_page_url': 'https://example.com/bundle',
            'bounce_rate': 55,
            'avg_session_duration_seconds': 118,
            'spend': 85,
            'atc': 7,
            'ic': 4,
            'orders': 1,
            'revenue': 3100
        },
    ])
