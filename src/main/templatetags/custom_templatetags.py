from django import template

register = template.Library()

@register.filter
def index(lst: list, index: int):
    try:
        result = lst[int(index)]
        return result
    except:
        return ""

@register.filter
def limit(text: str, max_len: int):
    return text[:max_len]


@register.filter
def short(text: str, max_len: int):
    ptn = "..."
    if len(text) > max_len:
        length = max_len - len(ptn)
        if max_len < 0:
            short_text = text[:max_len]
        else:
            short_text = text[:length] + ptn

        return short_text
    else:
        return text
