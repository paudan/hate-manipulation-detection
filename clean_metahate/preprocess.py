import string
import re
import ftfy
import emoji
import datefinder
from dateutil.parser import UnknownTimezoneWarning
import warnings

warnings.filterwarnings('ignore', category=UnknownTimezoneWarning)


def preprocess_text(text):
    # First, use ftfy to fix any encoding issues
    if hasattr(text, '__len__'):
        text = ftfy.fix_text(text)

        # Remove dates
        text = remove_dates(text)
        
        # Custom rules for punctuation fixing
        rules = [
            # Remove http and https links
            (r'https?://\S+', ''),
            # Remove mentions
            (r'@\S+', ''),
            # Remove tags
            (r'#\S+', ''),
            # Remove IPs
            (r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ''),
            # Remove NEWLINE_TOKEN
            ('NEWLINE_TOKEN', ''),
            # Remove consecutive repetitive punctuation, but keep a maximum of two for emphasis (e.g., !!)
            (r'([,\.?!])\1{2,}', r'\1\1'),
            # Normalize text by removing multiple repetitions of the same character: rulezzzz
            (r'([\w])\1{2,}', r'\1\1'),
            # Add space after comma, period, question mark, or exclamation mark if not followed by space
            (r'([,\.?!])(?=[^\s])', r'\1 '),
            # Remove space before comma, period, question mark, or exclamation mark
            (r'\s+([,\.?!])', r'\1'),
            # Fix multiple spaces
            (r'\s{2,}', ' '),
            # Ensure numbers have space before and after, except when punctuation or hyphen follows
            # Disable rule, as hacker language might be used: N4z!s 4r3 n0t n0rm4l
            # Does not support float values
            # (r'(\d)(?=[^\s\d,\.?!-])', r'\1 '),
            # (r'(?<=[^\s\d-])(\d)', r' \1')
        ]
    
        # Replace emoji with :shortcode:
        text = emoji.demojize(text, delimiters=(" ::", ":: "))
        # text = emoji.replace_emoji(text)
    
        # Apply each rule
        for pattern, replacement in rules:
            text = re.sub(pattern, replacement, text)

        stripped = string.punctuation + string.whitespace
        # :shortcode: ends with :, exclude it
        stripped = stripped.replace(":", "")
        text = text.strip(stripped)
    else:
        text = ''
    return text

def remove_dates(text):
    matches = datefinder.find_dates(text, source=True)
    replaced = []
    for match in matches:
        replaced.append(match[1])
    for replacement in replaced:
        text = text.replace(replacement, '')
    return text