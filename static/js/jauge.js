
 
var myChart_1 = echarts.init(document.getElementById('container_1'));
option = {
    tooltip: {
        formatter: '{a} <br/>{b} : {c}'
    },
    toolbox: {
        feature: {
            restore: {},
            saveAsImage: {}
        }
    },
    series: [
        {
            min:100,
            max: 266,
            radius: '80%',
            name: 'Temps de test',
            type: 'gauge',
            detail: {formatter: '{value}'},
            data: [{value: 101, name: 'Prediction'}]
        }
    ]
};
myChart_1.setOption(option);

