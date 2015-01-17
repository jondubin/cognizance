$(document).ready(function (){
    $(document).ready(function (){
        $("ul.nav > li > a").click(function (e){
            e.preventDefault();
            var this_elem = $(this);
            //$(this).animate(function(){
                $('html, body').animate({
                    scrollTop: $(this_elem.attr('href')).offset().top
                }, 400);
            //});
        });
    });
});