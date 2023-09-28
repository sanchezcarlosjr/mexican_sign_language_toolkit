import re

def tokenize(regex):
    stack = ""
    def match(text):
        nonlocal stack
        matcher = re.match(regex, text)
        if matcher and matcher.lastgroup == "noise":
            return None
        if matcher:
            stack = ""
            return (matcher.lastgroup, matcher.group())
        stack = stack + text
        matcher = re.search(regex, stack)
        if matcher:
            stack = ""
            return (matcher.lastgroup, matcher.group())
    return match