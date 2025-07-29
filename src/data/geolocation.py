from typing import Optional, Dict, Any
import ipaddress

class IPRiskEvaluator:
    def __init__(self):
        self.country_mapping = {}  # Would be loaded from database in production
    
    def is_foreign(self, ip_address: str, user_country: str = 'US') -> int:
        """Check if IP is foreign to user's country"""
        try:
            ip_country = self._get_country_from_ip(ip_address)
            return int(ip_country != user_country) if ip_country else 0
        except Exception:
            return 0
    
    def _get_country_from_ip(self, ip_address: str) -> Optional[str]:
        """Mock implementation - replace with real IP lookup"""
        try:
            if ipaddress.ip_address(ip_address).is_private:
                return None
            last_octet = int(ip_address.split('.')[-1])
            if last_octet < 64: return 'US'
            elif last_octet < 128: return 'CA'
            elif last_octet < 192: return 'UK'
            else: return 'CN'
        except ValueError:
            return None
    
    def evaluate(self, ip_address: str) -> Dict[str, Any]:
        """Comprehensive IP evaluation"""
        country = self._get_country_from_ip(ip_address)
        return {
            'ip_country': country or 'unknown',
            'is_foreign_ip': self.is_foreign(ip_address),
            'ip_risk_score': self._calculate_risk_score(country)
        }
    
    def _calculate_risk_score(self, country: Optional[str]) -> float:
        """Calculate risk score based on country"""
        if country in ['US', 'CA', 'UK']: return 0.1
        elif country in ['CN', 'RU']: return 0.7
        else: return 0.5

# Create alias for backward compatibility
IPToCountry = IPRiskEvaluator

# Explicitly expose both classes
__all__ = ['IPRiskEvaluator', 'IPToCountry']