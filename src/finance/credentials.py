#!/usr/bin/env python3
"""
Exchange Credentials Management

Provides a unified way to handle exchange credentials from either:
1. Environment variables (default, backward-compatible)
2. Direct dictionary/JSON configuration (for multi-worker setups)

This module enables running multiple trading bots with different API keys
in parallel without conflicting environment variable usage.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import os

from dotenv import load_dotenv


@dataclass
class ExchangeCredentials:
    """
    Container for exchange API credentials.
    
    Attributes:
        api_key: API key for the exchange
        secret_key: Secret key for the exchange
        demo_mode: Whether to use testnet/demo mode
        exchange_name: Name of the exchange (e.g., 'kraken', 'binance')
    """
    api_key: str
    secret_key: str
    demo_mode: bool = True
    exchange_name: str = "kraken"
    
    def __post_init__(self):
        """Validate credentials after initialization."""
        if not self.api_key:
            raise ValueError("api_key cannot be empty")
        if not self.secret_key:
            raise ValueError("secret_key cannot be empty")
        self.exchange_name = self.exchange_name.lower()
    
    @classmethod
    def from_env(cls, exchange_name: str = None) -> "ExchangeCredentials":
        """
        Create credentials from environment variables.
        
        This is the default method for backward compatibility with the
        existing .env file configuration.
        
        Args:
            exchange_name: Exchange name (default: from EXCHANGE_NAME env var)
            
        Returns:
            ExchangeCredentials instance
            
        Raises:
            ValueError: If required environment variables are missing
            
        Example:
            >>> creds = ExchangeCredentials.from_env()
            >>> creds = ExchangeCredentials.from_env("binance")
        """
        try:
            load_dotenv()
        except Exception:
            pass
        
        if exchange_name is None:
            exchange_name = os.getenv("EXCHANGE_NAME", "kraken").lower()
        else:
            exchange_name = exchange_name.lower()
        
        exchange_upper = exchange_name.upper()
        api_key_var = f"{exchange_upper}_API_KEY"
        secret_key_var = f"{exchange_upper}_SECRET_KEY"
        demo_var = f"{exchange_upper}_DEMO"
        
        api_key = os.getenv(api_key_var)
        secret_key = os.getenv(secret_key_var)
        
        if not api_key or not secret_key:
            missing = []
            if not api_key:
                missing.append(api_key_var)
            if not secret_key:
                missing.append(secret_key_var)
            raise ValueError(
                f"Missing required environment variables: {missing}\n"
                f"Please set:\n"
                f"  export {api_key_var}='your_api_key'\n"
                f"  export {secret_key_var}='your_secret_key'"
            )
        
        demo_mode = os.getenv(demo_var, "true").lower() == "true"
        
        return cls(
            api_key=api_key,
            secret_key=secret_key,
            demo_mode=demo_mode,
            exchange_name=exchange_name,
        )
    
    @classmethod
    def from_dict(cls, config: Dict) -> "ExchangeCredentials":
        """
        Create credentials from a dictionary (e.g., from JSON config).
        
        This method is used for multi-worker configurations where each
        worker has its own credentials specified in a JSON file.
        
        Args:
            config: Dictionary with keys:
                - api_key: API key (required)
                - secret_key: Secret key (required)
                - demo_mode: Boolean for testnet mode (optional, default: True)
                - exchange_name: Exchange name (optional, default: 'kraken')
                
        Returns:
            ExchangeCredentials instance
            
        Raises:
            ValueError: If required keys are missing
            
        Example:
            >>> config = {
            ...     "api_key": "KEY123",
            ...     "secret_key": "SECRET456",
            ...     "demo_mode": True,
            ...     "exchange_name": "kraken"
            ... }
            >>> creds = ExchangeCredentials.from_dict(config)
        """
        api_key = config.get("api_key")
        secret_key = config.get("secret_key")
        
        if not api_key or not secret_key:
            missing = []
            if not api_key:
                missing.append("api_key")
            if not secret_key:
                missing.append("secret_key")
            raise ValueError(f"Missing required keys in config: {missing}")
        
        return cls(
            api_key=api_key,
            secret_key=secret_key,
            demo_mode=config.get("demo_mode", True),
            exchange_name=config.get("exchange_name", "kraken"),
        )
    
    def to_dict(self) -> Dict:
        """
        Convert credentials to dictionary (without exposing full secrets).
        
        Useful for logging/debugging - shows masked keys.
        
        Returns:
            Dictionary with masked credentials
        """
        return {
            "api_key": f"{self.api_key[:4]}...{self.api_key[-4:]}" if len(self.api_key) > 8 else "****",
            "secret_key": "********",
            "demo_mode": self.demo_mode,
            "exchange_name": self.exchange_name,
        }
    
    def __repr__(self) -> str:
        """Safe string representation that doesn't expose secrets."""
        masked_key = f"{self.api_key[:4]}..." if len(self.api_key) > 4 else "****"
        return (
            f"ExchangeCredentials("
            f"api_key='{masked_key}', "
            f"demo_mode={self.demo_mode}, "
            f"exchange_name='{self.exchange_name}')"
        )


def get_default_credentials(exchange_name: str = None) -> ExchangeCredentials:
    """
    Get credentials using the default method (environment variables).
    
    Convenience function that wraps ExchangeCredentials.from_env().
    
    Args:
        exchange_name: Exchange name (optional)
        
    Returns:
        ExchangeCredentials instance
    """
    return ExchangeCredentials.from_env(exchange_name)


def validate_credentials(credentials: ExchangeCredentials) -> bool:
    """
    Validate that credentials are properly formed.
    
    Args:
        credentials: ExchangeCredentials instance to validate
        
    Returns:
        True if credentials appear valid
        
    Note:
        This only validates format, not whether the credentials
        actually work with the exchange.
    """
    if not credentials.api_key or len(credentials.api_key) < 8:
        return False
    if not credentials.secret_key or len(credentials.secret_key) < 8:
        return False
    if credentials.exchange_name not in ["kraken", "binance", "binanceus", "bybit"]:
        return False
    return True

