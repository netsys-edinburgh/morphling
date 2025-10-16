ID=$1
FLOPS=$2
MEMORY=$3
UL_BW=$4
DL_BW=$5
UL_LAT=$6
DL_LAT=$7
BACKEND=$8

# Hardcoded Redis host (host:port)
REDIS_HOST=${9:-127.0.0.1:6379}


morphling_device --id $ID --flops $FLOPS --memory $MEMORY --ul_bw $UL_BW --dl_bw $DL_BW --ul_lat $UL_LAT --dl_lat $DL_LAT --redis_host $REDIS_HOST --backend $BACKEND --cfg /app/config/proxy/cli.ini &
