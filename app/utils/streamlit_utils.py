import streamlit.components.v1 as components


def scroll_to_bottom():
    """Force le scroll vers le bas de la page"""
    components.html(
        """
        <script>
        setTimeout(function() {
            window.parent.document.querySelector('.main').scrollTo({
                top: window.parent.document.querySelector('.main')
                    .scrollHeight,
                behavior: 'smooth'
            });
        }, 100);
        </script>
        """,
        height=0,
    )
