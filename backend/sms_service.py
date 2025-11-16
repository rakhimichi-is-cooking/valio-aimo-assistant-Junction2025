"""
SMS Service using Twilio for shortage notifications
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Twilio setup
TWILIO_AVAILABLE = False
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    print("⚠️  Twilio not installed. Install with: pip install twilio python-dotenv")


class SMSService:
    """Send SMS notifications about shortages using Twilio"""

    def __init__(self):
        self.enabled = False

        if not TWILIO_AVAILABLE:
            print("❌ Twilio not available")
            return

        # Read credentials from environment
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_PHONE_NUMBER", "+15415322772")

        if not account_sid or not auth_token:
            print("❌ Twilio credentials not found in .env file")
            print("   Add TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN to .env")
            return

        try:
            self.client = Client(account_sid, auth_token)
            self.enabled = True
            print(f"✅ SMS Service initialized (from: {self.from_number})")
        except Exception as e:
            print(f"❌ Failed to initialize Twilio: {e}")

    def send_shortage_alert(self,
                           to_number: str,
                           customer_name: str,
                           shortages: List[Dict],
                           include_substitutes: bool = True) -> Dict:
        """
        Send SMS about product shortages

        Args:
            to_number: Phone number to send to (e.g., "+358442605413")
            customer_name: Customer name
            shortages: List of shortage events
            include_substitutes: Whether to mention replacements

        Returns:
            {'success': bool, 'message_sid': str, 'error': str}
        """
        if not self.enabled:
            return {
                'success': False,
                'error': 'SMS service not enabled. Check Twilio credentials.'
            }

        # Build message
        product_list = []
        for shortage in shortages[:5]:  # Limit to 5 products
            sku = shortage.get('sku', 'Unknown')
            product_name = shortage.get('product_name', 'Product')
            product_list.append(f"{product_name} ({sku})")

        products_text = ", ".join(product_list)

        if len(shortages) > 5:
            products_text += f", and {len(shortages) - 5} more"

        message_body = f"Hi {customer_name}, we have detected potential shortages on: {products_text}."

        if include_substitutes:
            message_body += " Would you like replacement suggestions?"

        try:
            message = self.client.messages.create(
                body=message_body,
                from_=self.from_number,
                to=to_number
            )

            return {
                'success': True,
                'message_sid': message.sid,
                'body': message_body
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def send_custom_message(self, to_number: str, message: str) -> Dict:
        """
        Send custom SMS message

        Args:
            to_number: Phone number
            message: Message text

        Returns:
            {'success': bool, 'message_sid': str, 'error': str}
        """
        if not self.enabled:
            return {
                'success': False,
                'error': 'SMS service not enabled'
            }

        try:
            msg = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )

            return {
                'success': True,
                'message_sid': msg.sid
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


# Global instance
_sms_service = None

def get_sms_service() -> SMSService:
    """Get singleton SMS service instance"""
    global _sms_service
    if _sms_service is None:
        _sms_service = SMSService()
    return _sms_service
