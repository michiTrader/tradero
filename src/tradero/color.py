from pintar.colors import HEX

def _get_logger_color_config(log_name_color: HEX, shade, sat):
    color_config = {
        'PERF': {
            'name': {
                'fore_color': log_name_color.shade(shade).saturate(sat), 
                'style':1}
        }, 
        'DEBUG': {
            'name': {
                'fore_color': log_name_color.shade(shade).saturate(sat), 
                'style':1
            }
        }, 
        'SIGNAL': {
            'name': {
                'fore_color': log_name_color, 
                'style':1
            }
        },
        'INFO': {
            'name': {
                'fore_color': log_name_color.shade(shade).saturate(sat), 
                'style':1
                }
            },
        'TRADING': {
            'name': {
                'fore_color': log_name_color, 
                'style':1
            }, 
            'levelname':{
                'forecolor':'#ffffff'
                }
            },
        'WARNING': {
            'name': {
                'fore_color': log_name_color, 
                'style':1
            }
        },
        'ERROR': {
            'name': {
                'fore_color': log_name_color, 
                'style':1
            }
        },
    }
    return color_config
    
