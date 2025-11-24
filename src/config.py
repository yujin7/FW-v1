"""
Configuration module for FW-v1 Crash Prediction System.
Loads API credentials from environment variables.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class QuantConnectConfig:
    """QuantConnect API configuration."""
    user_id: str = os.getenv("QUANTCONNECT_USER_ID", "")
    token: str = os.getenv("QUANTCONNECT_TOKEN", "")

    @property
    def is_configured(self) -> bool:
        return bool(self.user_id and self.token)


@dataclass
class OpenAIConfig:
    """OpenAI API configuration."""
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = os.getenv("OPENAI_MODEL", "gpt-4")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass
class HuggingFaceConfig:
    """Hugging Face API configuration."""
    token: str = os.getenv("HUGGINGFACE_TOKEN", "")

    @property
    def is_configured(self) -> bool:
        return bool(self.token)


@dataclass
class Config:
    """Main configuration container."""
    quantconnect: QuantConnectConfig
    openai: OpenAIConfig
    huggingface: HuggingFaceConfig

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from environment."""
        return cls(
            quantconnect=QuantConnectConfig(),
            openai=OpenAIConfig(),
            huggingface=HuggingFaceConfig(),
        )

    def validate(self) -> dict[str, bool]:
        """Check which APIs are configured."""
        return {
            "quantconnect": self.quantconnect.is_configured,
            "openai": self.openai.is_configured,
            "huggingface": self.huggingface.is_configured,
        }


# Global config instance
config = Config.load()


if __name__ == "__main__":
    # Test configuration
    print("Configuration Status:")
    for api, status in config.validate().items():
        print(f"  {api}: {'✓ Configured' if status else '✗ Missing'}")
